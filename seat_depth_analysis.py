import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import mediapipe as mp
import json
from datetime import datetime
import time
import os

def process_seat_depth_analysis(image_path, eye_to_ear_cm=7.0, sam_checkpoint="sam_vit_b_01ec64.pth"):
    """
    Main function to process seat depth analysis
    
    Args:
        image_path: Path to the input image
        eye_to_ear_cm: Real-world eye to ear distance for scaling (default 7.0 cm)
        sam_checkpoint: Path to SAM model checkpoint
    
    Returns:
        tuple: (output_json, pose_image, seat_band_image, final_image)
    """
    start_time = time.time()

    def put_text_safe(image, text, org, font, font_scale, color, thickness):
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        x, y = org
        h, w = image.shape[:2]

        # Adjust X if text goes out on the right
        if x + text_width > w:
            x = w - text_width - 5  # 5 pixel padding from right

        # Adjust X if text goes out on the left
        if x < 0:
            x = 5  # 5 pixel padding from left

        # Adjust Y if text goes above image
        if y - text_height < 0:
            y = text_height + 5  # push down

        # Adjust Y if text goes below image
        if y > h:
            y = h - 5

        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

        
    # === Load image ===
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # === Run MediaPipe Pose Detection ===
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError("No pose detected in the image")

    landmarks = results.pose_landmarks.landmark

    # === Get Knee and Eye X,Y coordinates ===
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    left_eye = landmarks[2]
    right_eye = landmarks[5]
    right_ear = landmarks[8]
    left_ear = landmarks[7]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    # Convert to pixel coordinates
    left_knee_px = (int(left_knee.x * w), int(left_knee.y * h))
    right_knee_px = (int(right_knee.x * w), int(right_knee.y * h))
    left_eye_px = (int(left_eye.x * w), int(left_eye.y * h))
    right_eye_px = (int(right_eye.x * w), int(right_eye.y * h))
    left_ear_px = (int(left_ear.x * w), int(left_ear.y * h))
    right_ear_px = (int(right_ear.x * w), int(right_ear.y * h))
    left_hip_px = (int(left_hip.x * w), int(left_hip.y * h))
    right_hip_px = (int(right_hip.x * w), int(right_hip.y * h))

    # === Determine Facing Direction ===
    avg_knee_x = (left_knee_px[0] + right_knee_px[0]) / 2
    avg_eye_x = (left_eye_px[0] + right_eye_px[0]) / 2
    facing_direction = "right" if avg_knee_x > avg_eye_x else "left"

    # === Create Pose Overlay (Image 1) ===
    pose_image = image_rgb.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    mp_drawing.draw_landmarks(
        pose_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # === Step 1: Detect Chair with YOLOv8 ===
    yolo_model = YOLO("yolov8n.pt")
    yolo_results = yolo_model(image_rgb)

    # === Step 2: Get Chair Box ===
    chair_box = None
    chair_confidence = 0.0
    for result in yolo_results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if int(cls.item()) == 56:  # 56 = chair
                chair_box = box.cpu().numpy().astype(int)
                chair_confidence = float(conf.item())
                break

    if chair_box is None:
        raise ValueError("No chair detected in the image")

    x1, y1, x2, y2 = chair_box
    chair_height = y2 - y1
    adjusted_y1 = y1 + int(0.25 * chair_height)
    input_box = np.array([x1, adjusted_y1, x2, y2])

    # === Step 3: Load SAM ===
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # === Step 4: Predict Mask from Bounding Box ===
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=True)
    best_mask = masks[np.argmax(scores)]

    # === Step 5: Largest Component Only ===
    def get_largest_connected_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return mask
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return labels == largest_label

    cleaned_mask = get_largest_connected_component(best_mask)

    # === Step 6: Estimate Seat Front ===
    knee_y = int((left_knee_px[1] + right_knee_px[1]) / 2)
    band_thickness = chair_height // 2
    y_min = max(0, knee_y - band_thickness)
    y_max = min(h, knee_y + band_thickness)
    band = cleaned_mask[y_min:y_max, :]
    chair_pixels_x = np.where(band)[1]

    if chair_pixels_x.size == 0:
        raise ValueError("No chair pixels detected at knee level")

    seat_front_x = chair_pixels_x.max() if facing_direction == "right" else chair_pixels_x.min()
    seat_front_y = knee_y

    # === Create Seat Front Band Visualization (Image 2) ===
    seat_band_image = image_rgb.copy()
    cv2.line(seat_band_image, (0, y_min), (w, y_min), (0, 255, 0), 2)
    cv2.line(seat_band_image, (0, y_max), (w, y_max), (0, 255, 0), 2)
    cv2.circle(seat_band_image, (seat_front_x, seat_front_y), 8, (0, 0, 255), -1)
    put_text_safe(seat_band_image, "Seat Front", (seat_front_x + 10, seat_front_y - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # === Calculate Back of Knee Position ===
    def euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Calculate thigh length for proportional offset
    if facing_direction == "right":
        hip_pt = right_hip_px
        knee_pt_original = right_knee_px
    else:
        hip_pt = left_hip_px
        knee_pt_original = left_knee_px

    thigh_length_px = euclidean_distance(hip_pt, knee_pt_original)

    # Back of knee is typically 12-15% of thigh length behind knee center
    back_of_knee_offset = thigh_length_px * 0.13  # 13% of thigh length

    # Apply offset in the backward direction
    if facing_direction == "right":
        knee_pt = (int(knee_pt_original[0] - back_of_knee_offset), knee_pt_original[1])
    else:
        knee_pt = (int(knee_pt_original[0] + back_of_knee_offset), knee_pt_original[1])

    # === Calculate Measurements ===
    clearance_px = abs(seat_front_x - knee_pt[0])

    # Check visibility and calculate eye-to-ear distance
    visibility_warnings = []
    if facing_direction == "right" and (right_eye.visibility < 0.5 or right_ear.visibility < 0.5):
        visibility_warnings.append("Right eye or ear not clearly visible. Scaling may be inaccurate.")
    elif facing_direction == "left" and (left_eye.visibility < 0.5 or left_ear.visibility < 0.5):
        visibility_warnings.append("Left eye or ear not clearly visible. Scaling may be inaccurate.")

    if facing_direction == "right":
        eye_coord = right_eye_px
        ear_coord = right_ear_px
    else:
        eye_coord = left_eye_px
        ear_coord = left_ear_px

    eye_to_ear_px = euclidean_distance(eye_coord, ear_coord)
    pixels_per_cm = eye_to_ear_px / eye_to_ear_cm
    clearance_cm = clearance_px / pixels_per_cm

    # Determine if back of knee is behind seat front
    if facing_direction == "right":
        knee_behind_seat = knee_pt[0] < seat_front_x
    else:
        knee_behind_seat = knee_pt[0] > seat_front_x

    # === Classification ===

    category = "Too Short"
    if knee_behind_seat or clearance_cm < 2:
        if clearance_cm < 2:
            category = "Too Deep"
            reasoning = f"Clearance of {clearance_cm:.2f}cm is less than 2cm minimum"
        elif knee_behind_seat:
            category = "Too Deep"
            reasoning = "Back of knee is behind seat front"
    elif clearance_cm <= 6:
        category = "Optimal"
        reasoning = f"Clearance of {clearance_cm:.2f}cm falls within optimal range (2-6cm)"
    else:
        category = "Too Short"
        reasoning = f"Clearance of {clearance_cm:.2f}cm exceeds 6cm optimal maximum"

    # === Create Final Visualization (Image 3) ===
    final_image = image_rgb.copy()

    # Draw seat front and knee
    cv2.circle(final_image, (seat_front_x, seat_front_y), 8, (0, 0, 255), -1)
    cv2.circle(final_image, knee_pt, 8, (255, 0, 0), -1)

    # Height at which the line floats
    line_y = min(seat_front_y, knee_pt[1]) - 30

    # Draw horizontal line (floating)
    cv2.line(final_image, (min(seat_front_x, knee_pt[0]), line_y), 
                   (max(seat_front_x, knee_pt[0]), line_y), 
             (255, 255, 0), 2)

    # Add arrow tips
    cv2.arrowedLine(final_image,
                    (min(seat_front_x, knee_pt[0]) + 20, line_y),
                    (min(seat_front_x, knee_pt[0]), line_y),
                    (255, 255, 0), 2, tipLength=0.4)

    cv2.arrowedLine(final_image,
                    (max(seat_front_x, knee_pt[0]) - 20, line_y),
                    (max(seat_front_x, knee_pt[0]), line_y),
                    (255, 255, 0), 2, tipLength=0.4)

    # Put clearance text above the line
    put_text_safe(final_image, f"Knee clearance: {clearance_cm:.1f} cm", 
              (min(seat_front_x, knee_pt[0]) + 10, line_y - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw eye-to-ear line
    cv2.line(final_image, eye_coord, ear_coord, (0, 255, 0), 2)
    put_text_safe(final_image, f"{eye_to_ear_cm:.1f}cm", 
              (eye_coord[0], eye_coord[1] - 10), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === Generate JSON Output ===
    processing_time = int((time.time() - start_time) * 1000)
    
    output_json = {
       "frame_id": os.path.basename(image_path),
        "timestamp": datetime.now().isoformat(),
        "pose_detection": {
            "pose_detected": True,
            "facing_direction": facing_direction,
            "landmarks_visibility": {
                "left_eye": float(left_eye.visibility),
                "right_eye": float(right_eye.visibility),
                "left_ear": float(left_ear.visibility),
                "right_ear": float(right_ear.visibility),
                "left_knee": float(left_knee.visibility),
                "right_knee": float(right_knee.visibility),
                "left_hip": float(left_hip.visibility),
                "right_hip": float(right_hip.visibility)
            }
        },
        "chair_detection": {
            "chair_detected": True,
            "chair_bbox": chair_box.tolist(),
            "chair_confidence": chair_confidence
        },
        "measurements": {
            "eye_to_ear_distance_px": float(eye_to_ear_px),
            "eye_to_ear_distance_cm": float(eye_to_ear_cm),
            "pixels_per_cm": float(pixels_per_cm),
            "seat_front_position": [int(seat_front_x), int(seat_front_y)],
            "back_of_knee_position": [int(knee_pt[0]), int(knee_pt[1])],
            "knee_clearance_px": float(clearance_px),
            "knee_clearance_cm": float(clearance_cm),
            "thigh_length_px": float(thigh_length_px),
            "back_of_knee_offset_applied": float(back_of_knee_offset)
        },
        "classification": {
            "category": category,
            "knee_behind_seat": bool(knee_behind_seat),
            "reasoning": reasoning
        },
        "debug_info": {
            "band_y_range": [int(y_min), int(y_max)],
            "chair_pixels_detected": int(chair_pixels_x.size),
            "segmentation_success": True,
            "scaling_method": "eye_to_ear_reference"
        },
        "warnings": visibility_warnings,
        "processing_time_ms": processing_time
    }

    return output_json, pose_image, seat_band_image, final_image