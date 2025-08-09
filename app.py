import streamlit as st
import cv2
import numpy as np
import json
from PIL import Image
import io
import tempfile
import os
from datetime import datetime
from seat_depth_analysis import process_seat_depth_analysis
st.set_page_config(
        page_title="Seat Depth Analyzer",
        page_icon="🪑",
        layout="wide",
        initial_sidebar_state="expanded"
    )
# Custom CSS for background gradient and styling
st.markdown("""
    <style>
    /* Gradient background for the whole app */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #ffffff, #fce4ec);
        background-attachment: fixed;
    }

    /* Make metric cards look modern */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Beautify sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f8e9, #ffffff);
    }

    /* Make headers and titles prettier */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }

    /* Button tweaks */
    button[kind="primary"] {
        background-color: #00796b;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }

    button[kind="primary"]:hover {
        background-color: #004d40;
        color: white;
    }

    /* Download button */
    div.stDownloadButton > button {
        background-color: #3949ab;
        color: white;
        border-radius: 8px;
    }

    div.stDownloadButton > button:hover {
        background-color: #1a237e;
    }

    </style>
""", unsafe_allow_html=True)

def main():
   
    
    st.title("🪑✨ SitSmart")
    st.subheader("Analyze your seat — because not all thrones are ergonomic :)")

    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("📐 Configuration: Anthropometric Assumption")
    st.sidebar.markdown(
    "We assume a default **ear-to-eye distance of 7 cm**, based on average adult anatomy. "
    "You may change this value if the subject in the image deviates significantly."
    )
    st.sidebar.caption("Don’t worry, no need to measure your face with a ruler. 📏👂")



    # Eye-to-ear distance setting
    eye_to_ear_cm = st.sidebar.slider(
        "Eye-to-Ear Distance (cm)",
        min_value=5.0,
        max_value=10.0,
        value=7.0,
        step=0.1,
        help="Average distance from eye to ear for scaling reference (default: 7.0 cm)"
    )
    
    sam_checkpoint = "sam_vit_b_01ec64.pth"

    # Information section
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Classification Guide")
    st.sidebar.markdown("""
    **Optimal**: 2-6 cm clearance from seat front to back of knee
    
    **Too Deep**: Less than 2 cm clearance (circulation risk)
    
    **Too Short**: More than 6 cm clearance (poor thigh support)
    """)
    
    
    st.header("📤 Choose Image")
    
    # Image source selection
    image_source = st.radio(
        "Select image source:",
        options=["Upload your own", "Choose from samples"],
        horizontal=True
    )
    
    selected_image_path = None
    uploaded_file = None
    
    if image_source == "Upload your own":
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a side-profile image of person seated on chair",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload a clear side-profile image showing the person seated with their back against the chair"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=500)

    else:  # Choose from samples
        sample_category = st.selectbox(
            "Select sample category:",
            options=["optimal", "too_deep", "too_short"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Get available sample images
        sample_images = get_sample_images(sample_category)
        
        if sample_images:
            selected_image = st.selectbox(
                "Select sample image:",
                options=sample_images,
                format_func=lambda x: x.replace("_", " ").replace(".png", "").replace(".jpg", "").replace(".jpeg", "").title()
            )
            
            selected_image_path = os.path.join("sample_images", sample_category, selected_image)
            
            if os.path.exists(selected_image_path):
                # Display selected sample image
                image = Image.open(selected_image_path)
                st.image(image, caption=f"Sample: {selected_image}", width=500)
            else:
                st.error(f"Sample image not found: {selected_image_path}")
                selected_image_path = None
        else:
            st.warning(f"No sample images found in sample_images/{sample_category}/")
    
    # Process button
    if (uploaded_file is not None or selected_image_path is not None):
        if st.button("🔍 Analyze Seat Depth", type="primary"):
            if image_source == "Upload your own":
                process_uploaded_image(uploaded_file, eye_to_ear_cm, sam_checkpoint)
            else:
                process_sample_image(selected_image_path, eye_to_ear_cm, sam_checkpoint)

    st.info("Upload an image and click 'Analyze Seat Depth' to see results here.")

def get_sample_images(category):
    """Get list of sample images for a given category"""
    sample_dir = os.path.join("sample_images", category)
    
    if not os.path.exists(sample_dir):
        return []
    
    # Get all image files
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    sample_images = []
    
    try:
        for file in os.listdir(sample_dir):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                sample_images.append(file)
        return sorted(sample_images)  # Sort alphabetically
    except Exception:
        return []

def process_uploaded_image(uploaded_file, eye_to_ear_cm, sam_checkpoint):
    """Process the uploaded image and display results"""
    

    with st.spinner("🔄 Processing uploaded image..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            # Process the image using your main function
            output_json, pose_image, seat_band_image, final_image = process_seat_depth_analysis(
                temp_path, eye_to_ear_cm, sam_checkpoint
            )
            
            # Display results
            display_results(output_json, pose_image, seat_band_image, final_image)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.error("Please ensure the image shows a clear side profile of a person seated on a chair.")

def process_sample_image(image_path, eye_to_ear_cm, sam_checkpoint):
    """Process the sample image and display results"""
    

    with st.spinner("🔄 Processing sample image..."):
        try:
            # Process the image using your main function
            output_json, pose_image, seat_band_image, final_image = process_seat_depth_analysis(
                image_path, eye_to_ear_cm, sam_checkpoint
            )
            
            # Display results with sample info
            st.info(f"📁 **Sample Image**: {os.path.basename(image_path)} from {os.path.basename(os.path.dirname(image_path))} category")
            display_results(output_json, pose_image, seat_band_image, final_image)
            
        except Exception as e:
            st.error(f"❌ Error processing sample image: {str(e)}")
            st.error(f"Could not process: {image_path}")


def display_results(output_json, pose_image, seat_band_image, final_image):
    """Display the analysis results in the Streamlit interface"""
    
    st.header("📊 Analysis Results")

    # Classification result with color coding
    category = output_json['classification']['category']
    
    if category == "Optimal":
        st.success(f"✅ **Classification: {category}**")
    elif category == "Too Deep":
        st.error(f"🔴 **Classification: {category}**")
    else:  # Too Short
        st.warning(f"⚠️ **Classification: {category}**")
    
    # Key measurements
    st.markdown("### 📏 Key Measurements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clearance_cm = output_json['measurements']['knee_clearance_cm']
        st.metric(
            "Knee Clearance",
            f"{clearance_cm:.2f} cm",
            help="Distance between seat front and back of knee"
        )
    
    with col2:
        facing = output_json['pose_detection']['facing_direction']
        st.metric(
            "Facing Direction",
            facing.title(),
            help="Direction the person is facing in the image"
        )
    
    with col3:
        pixels_per_cm = output_json['measurements']['pixels_per_cm']
        st.metric(
            "Scale Factor",
            f"{pixels_per_cm:.2f} px/cm",
            help="Pixels per centimeter for measurements"
        )
    
    # Reasoning
    st.markdown("### 🤔 Analysis Reasoning")
    st.info(output_json['classification']['reasoning'])
    
    # Image results tabs
    st.markdown("### 🖼️ Analysis Visualization")
    
    tab1, tab2, tab3 = st.tabs(["Final Result", "Pose Detection", "Seat Band Analysis"])
    
    with tab1:
        st.image(
            final_image,
            caption="Final Analysis - Knee Clearance Measurement",
            width = 500
        )
        st.markdown("**Blue dot**: Seat front edge | **Red dot**: Back of knee position")

    with tab2:
        st.image(
            pose_image,
            caption="Pose Detection Overlay",
            width = 500
        )
        st.markdown("Shows detected pose landmarks and connections")
    
    with tab3:
        st.image(
            seat_band_image,
            caption="Seat Front Detection Band",
            width = 500
        )
        st.markdown("**Green lines**: Analysis band | **Blue dot**: Detected seat front")
    
    # Detailed measurements (expandable)
    with st.expander("📐 Detailed Measurements"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Measurements": {
                    "Knee Clearance (px)": f"{output_json['measurements']['knee_clearance_px']:.1f}",
                    "Knee Clearance (cm)": f"{output_json['measurements']['knee_clearance_cm']:.2f}",
                    "Eye-to-Ear Distance (px)": f"{output_json['measurements']['eye_to_ear_distance_px']:.1f}",
                    "Thigh Length (px)": f"{output_json['measurements']['thigh_length_px']:.1f}",
                    "Seat Front Position": output_json['measurements']['seat_front_position'],
                    "Back of Knee Position": output_json['measurements']['back_of_knee_position']
                }
            })
        
        with col2:
            st.json({
                "Detection Info": {
                    "Chair Detected": output_json['chair_detection']['chair_detected'],
                    "Chair Confidence": f"{output_json['chair_detection']['chair_confidence']:.3f}",
                    "Pose Detected": output_json['pose_detection']['pose_detected'],
                    "Processing Time": f"{output_json['processing_time_ms']} ms"
                }
            })
    
    # Warnings
    if output_json['warnings']:
        st.markdown("### ⚠️ Warnings")
        for warning in output_json['warnings']:
            st.warning(warning)
    
    # Download JSON results
    st.markdown("### 💾 Download Results")
    
    json_str = json.dumps(output_json, indent=2)
    st.download_button(
        label="📄 Download JSON Report",
        data=json_str,
        file_name=f"seat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",  help="Yes, a whole JSON just for your seat."
    )


if __name__ == "__main__":
    # Add sample images section at the bottom
    
    
    # Footer
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Ergonomic Seat Depth Analyzer | Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )
    main()