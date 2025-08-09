[![Live Demo – Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/nitikaborkar/seat-depth-analyser)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Seat Depth Analyzer - Technical Documentation

🪑✨ Seat Depth Analyzer
An AI-powered computer vision application that analyzes ergonomic seating conditions from side-profile images and classifies seat pan depth as Optimal, Too Deep, or Too Short.

🚀 Quick Start

##  Online (No Installation Needed)
You can try the Seat Depth Analyzer instantly without installing anything:  
🔗 **[Click here to open the live app on Hugging Face Spaces](https://huggingface.co/spaces/nitikaborkar/seat-depth-analyser)**

Features of the online version:
- Upload your own image or choose from sample images
- Instantly see classification & clearance measurements
- Runs fully in the browser via Hugging Face Spaces

---

## 💻 Quick Start – Local Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/nitikaborkar/Seat_Pan_Depth_Classification.git
    cd Seat_Pan_Depth_Classification
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**
    ```bash
    streamlit run app.py
    ```
    *Note: The SAM model (`sam_vit_b_01ec64.pth`) is included in the submission.*

4. **Open in Browser**
    Navigate to: [http://localhost:8501](http://localhost:8501)

5. **Test the App**
    - Upload a side-profile image of someone seated, **or**
    - Try the included sample images
    - Click **"🔍 Analyze Seat Depth"**
## 🎯 Project Overview

The Seat Depth Analyzer is an AI-powered computer vision application that analyzes ergonomic seating conditions from side-profile images. It classifies seat pan depth as **Optimal**, **Too Deep**, or **Too Short** based on the clearance between the seat front edge and the back of the user's knee.

### Ergonomic Classification Criteria
- **Optimal**: 2-6 cm clearance (proper thigh support without circulation issues)
- **Too Deep**: <2 cm clearance or knee behind seat edge (circulation risk)
- **Too Short**: >6 cm clearance (insufficient thigh support)

---

## 🧠 Technical Architecture

### Multi-Model Pipeline
The solution employs a sophisticated multi-model approach combining three state-of-the-art computer vision models:

```
Input Image → Pose Detection → Chair Detection → Seat Segmentation → Measurement → Classification →  Output 
                        ↓              ↓               ↓               ↓                ↓              
                        MediaPipe     YOLOv8n        SAM (ViT-B)      CV Analysis    Ergonomic     
                        Pose           (Chair)      Segmentation     & Scaling          Rules        
    ```

---

## 🤖 Model Selection and Rationale

### 1. Pose Estimation Model Choice: MediaPipe Pose

**Why MediaPipe Pose?**
- **High Accuracy**: Proven performance on diverse body poses and lighting conditions
- **Landmark Precision**: Provides 33 precise body landmarks including knees, hips, eyes, and ears
- **Visibility Scoring**: Each landmark includes visibility confidence, crucial for side-profile analysis
- **Computational Efficiency**: Real-time performance suitable for web applications
- **Robustness**: Handles partial occlusion and varied clothing better than alternatives

**Alternative Considered**: OpenPose
- **Rejected because**: Higher computational requirements, less optimized for single-person detection
- **MediaPipe advantage**: Better integration with web deployment, more stable landmark tracking

**Key Landmarks Used**:
- **Knees** (left/right): Primary measurement points
- **Eyes/Ears**: Scaling reference (anatomical constant)
- **Hips**: Thigh length calculation for anatomical proportions

### 2. Chair Detection Model: YOLOv8n

**Why YOLOv8n?**
- **Speed vs. Accuracy Balance**: Nano version provides sufficient accuracy for chair detection while maintaining fast inference
- **Pre-trained COCO**: Chair class (ID: 56) readily available without custom training
- **Bounding Box Precision**: Accurate enough to constrain segmentation region
- **Memory Efficiency**: Suitable for deployment environments

**Usage Strategy**:
- Extract chair bounding box (which was then sent to SAM Meta Model)
- This was also used to Apply 25% vertical crop from top (focuses on seat area, excludes backrest)
- Use as region-of-interest for segmentation model

### 3. Segmentation Model: SAM (Segment Anything Model) ViT-B

**Why SAM?**
SAM has point based or bounding-box based or even prompt based segmentation ability
So I used it to mask out the chair from the image in order to be able to better focus on the seat pan front

- **Bounding Box-Based Segmentation**: Can segment objects using bounding box prompts
- **High-Quality Masks**: Superior edge precision compared to traditional segmentation
- **Generalization**: Works on furniture without specific training
- **Multi-Scale Features**: ViT-B provides good balance of accuracy and speed

**Alternative Considered**: Traditional edge detection + contour finding
- **Rejected because**: Poor performance on textured seats, lighting variations, and complex backgrounds
- **SAM advantage**: Semantic understanding of object boundaries

---

## 📐 Measurement Methodology

### Knee Position Estimation

**Challenge**: MediaPipe knee landmarks represent joint centers, not the back of the knee (popliteal area) needed for ergonomic measurement.

**Solution**: Anatomical Offset Calculation
```python
# Calculate thigh length for proportional offset
thigh_length_px = euclidean_distance(hip_position, knee_position)

# Back of knee offset: 13% of thigh length behind knee center
back_of_knee_offset = thigh_length_px * 0.13

# Apply directional offset based on facing direction
if facing_direction == "right":
    back_of_knee_x = knee_center_x - back_of_knee_offset
else:
    back_of_knee_x = knee_center_x + back_of_knee_offset
```

**Rationale for 13% Offset**:
- Since we need the back of the knee and not the knee (which MediaPipe landmark gives us )
- Based on anthropometric studies of knee anatomy - the back of the thigh would be approximately 12-15% offset from the knee
- Validated against manual measurements on test images
- Accounts for the distance from knee joint center to posterior knee surface

### Seat Edge Detection

**Multi-Step Process**:

1. **Region Extraction**:
   ```python
   # Create analysis band around knee level
   knee_y = average_knee_height
   band_thickness = chair_height // 2
   analysis_region = mask[knee_y - band_thickness : knee_y + band_thickness, :]
   ```

2. **Edge Detection Strategy**:
   - Extract chair mask pixels within the analysis band
   - Find extreme X-coordinate based on facing direction
   - **Right-facing**: Rightmost chair pixel (seat front)
   - **Left-facing**: Leftmost chair pixel (seat front)

3. **Validation**:
   - Ensure sufficient chair pixels detected in analysis region
   - Cross-validate with chair bounding box constraints

### Scaling and Real-World Measurements

Now that I had the back of the knee and also the seat front. I could calculate the distance in pixels. But this needed to be converted to cms for our problem statemet 

**Reference-Based Scaling**:
```python
# Use eye-to-ear distance as anatomical constant
eye_to_ear_distance_px = euclidean_distance(eye_landmark, ear_landmark)
eye_to_ear_distance_cm = 7.0  # Average adult measurement

pixels_per_cm = eye_to_ear_distance_px / eye_to_ear_distance_cm
clearance_cm = clearance_pixels / pixels_per_cm
```

**Why Eye-to-Ear Distance?**
- **Anatomical Constant**: Relatively consistent across adults (6.5-7.5 cm)
- **Visibility**: Usually visible in side-profile images
- **Stability**: Less affected by posture compared to other facial measurements

### Facing Direction Detection
- Determines if person faces left or right in image
    
Method: Compare average X-coordinates of knees vs. eyes
- If knees are right of eyes: facing right
- If knees are left of eyes: facing left

This affects:
1. Which knee/eye/ear to use for measurements
2. Direction of anatomical offsets
3. Seat edge detection logic

---

## 🚧 Challenges in Spacing Detection

### 1. Pose Detection Challenges

**Challenge**: Partial Occlusion
- **Problem**: Knees/hips may be obscured by desk, clothing, or shadows
- **Solution**: Visibility scoring and confidence thresholds
- **Mitigation**: Multi-landmark validation, graceful degradation

**Challenge**: Clothing Variations
- **Problem**: Baggy pants obscure actual knee position
- **Solution**: Anatomical offset based on skeletal landmarks rather than clothing contours
- **Limitation**: Still estimates through clothing, may introduce small errors

### 2. Chair Segmentation Challenges

**Challenge**: Complex Seat Materials
- **Problem**: Mesh, leather, fabric textures confuse edge detection
- **Solution**: SAM's semantic understanding handles material variations
- **Remaining Issue**: Highly reflective or transparent materials

**Challenge**: Partial Chair Visibility
- **Problem**: Desk, person's body may occlude seat edges
- **Solution**: Focus analysis on knee-level band where seat is most likely visible
- **Limitation**: Deep occlusion may cause detection failure

### 3. Scaling and Measurement Challenges

**Challenge**: Camera Perspective Distortion
- **Problem**: Non-perpendicular camera angles affect measurements
- **Solution**: Assume reasonable side-profile positioning
- **Limitation**: Extreme angles (>30°) may introduce errors

**Challenge**: Depth Perception in 2D Images
- **Problem**: Cannot measure true 3D distances
- **Solution**: Project measurements onto image plane
- **Assumption**: Person and chair are roughly in the same plane

### 4. Lighting and Image Quality

**Challenge**: Poor Lighting Conditions
- **Problem**: Shadows, backlighting affect landmark detection
- **Solution**: MediaPipe's robustness to lighting variations
- **Enhancement**: Preprocessing could include histogram equalization

---

## 🎯 Accuracy Improvement Suggestions

### Short-Term Improvements

1. **Enhanced Preprocessing**
    - Maybe can have improced contrast using certain methods like histogram equilization

2. **Multi-Reference Scaling**
   - Combine eye-to-ear with other facial measurements
   - Use hand/finger dimensions when visible
   - Cross-validate scaling factors

### Medium-Term Enhancements

1. **Custom Training Data**
   - Collect ergonomic seating dataset with ground truth measurements
   - Then we could actually fine-tune pose estimation on seated postures
   - And train a specialized chair segmentation model

2. **Multi-Frame Analysis**
   - Process video streams and have average measurements across multiple frames

3. **3D Pose Estimation**
   - Integrate depth estimation models
   - Calculate true 3D clearances

### Long-Term Research Directions

**Multi-Modal Sensing**
   - Combine computer vision with pressure sensors
   - Integrate with smart chair systems
   - Real-time posture monitoring

---


## 📊 Development Process and Design Decisions

### Iterative Development Approach

1. **Phase 1: Core Detection**
   - Implemented basic pose detection
   - Added simple chair detection
   - Established measurement pipeline

2. **Phase 2: Accuracy Enhancement**
   - Integrated SAM for precise segmentation
   - Added anatomical offset calculations
   - Implemented multi-scale analysis

3. **Phase 3: User Experience**
   - Built Streamlit interface
   - Added visualization pipeline
   - Implemented sample image system

4. **Phase 4: Robustness**
   - Enhanced error handling
   - Added confidence scoring
   - Implemented comprehensive testing

### Key Design Decisions

**Decision 1: Multi-Model vs. Single Model**
- **Chosen**: Multi-model pipeline
- **Rationale**: Each model excels in its domain (pose, detection, segmentation)
- **Trade-off**: Complexity vs. accuracy

**Decision 2: Real-time vs. Batch Processing**
- **Chosen**: Single image analysis
- **Rationale**: Simplicity, easier deployment
- **Future**: Could extend to video streams

**Decision 3: Cloud vs. Local Processing**
- **Chosen**: Local processing capability
- **Rationale**: Privacy, offline usage
- **Deployment**: Supports both local and cloud deployment

### Assumptions and Limitations

**Key Assumptions**:
1. **Side Profile View**: Person is photographed from the side 
2. **Seated Posture**: Back is against or near chair backrest
3. **Standard Chair**: Conventional office chair design
4. **Adult Subjects**: Eye-to-ear scaling appropriate for adults
5. **Static Analysis**: Single-moment analysis, not dynamic posture

**Known Limitations**:
1. **2D Analysis**: Cannot account for chair/body rotation out of image plane
2. **Clothing Effects**: Thick clothing may obscure true body landmarks
3. **Lighting Dependency**: Very poor lighting may affect landmark detection
4. **Chair Variety**: Unusual chair designs may confuse detection
5. **Anthropometric Variation**: Fixed scaling may not suit all body types

---

## 🔍 Validation and Testing Strategy

### Test Coverage

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end pipeline validation
3. **Accuracy Tests**: Ground truth comparison on sample images
4. **Edge Case Tests**: Handling of failure conditions
5. **Performance Tests**: Processing time benchmarking

### Sample Dataset

- **Optimal Cases (3 samples)**: Clear examples of proper seating
- **Too Deep Cases (4 samples)**: Various levels of excessive depth
- **Too Short Cases (8 samples)**: Range of insufficient depth scenarios
---

### Technical References
1. **MediaPipe Pose**: [Google Research Paper](https://arxiv.org/abs/2006.10204)
2. **SAM (Segment Anything)**: [Meta AI Research](https://arxiv.org/abs/2304.02643)
3. **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)

### Dataset and Tools
- **Sample Images**: Custom collected and validated
- **Development Environment**: Python 3.9, PyTorch, OpenCV
- **Deployment Platform**: Streamlit Cloud

### Anthropometric Data Sources
- **Eye-to-Ear Measurements**: Reference paper : "An anthropometric study to evaluate the correlation between the occlusal vertical dimension and length of the thumb" - Clinical, Cosmetic and Investigational Dentistry
