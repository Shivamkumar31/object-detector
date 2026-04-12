"""
Real-Time Object Detection Dashboard using YOLOv8
Streamlit Web Interface
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict
import time

# Set page config
st.set_page_config(
    page_title="Object Detection Dashboard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎥 Real-Time Object Detection Dashboard")
st.markdown("Powered by YOLOv8 | Detect objects in real-time")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_size = st.sidebar.selectbox(
    "Select YOLOv8 Model Size",
    ["nano", "small", "medium", "large", "xlarge"],
    help="nano=fastest, xlarge=most accurate"
)

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Only show detections with confidence > threshold"
)

# IoU threshold
iou_threshold = st.sidebar.slider(
    "IOU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Intersection Over Union for NMS"
)

# Input source
st.sidebar.header("📹 Input Source")
input_source = st.sidebar.radio(
    "Choose input type",
    ["📷 Webcam", "📹 Upload Video", "🖼️ Upload Image"]
)

# Load model (cached for performance)
# @st.cache_resource
# def load_model(model_size):
#     """Load YOLOv8 model"""
#     model_name = f"yolov8{model_size}.pt"
#     st.info(f"📥 Loading {model_name}... (one-time download)")
#     model = YOLO(model_name)
#     return model

# # Initialize model


@st.cache_resource
def load_model(model_size):
    """Load YOLOv8 model"""
    
    size_map = {
        "nano": "n",
        "small": "s",
        "medium": "m",
        "large": "l",
        "xlarge": "x"
    }
    
    model_name = f"yolov8{size_map[model_size]}.pt"
    
    st.info(f"📥 Loading {model_name}... (one-time download)")
    model = YOLO(model_name)
    
    return model

# # Initialize model
model = load_model(model_size)
# Process image/video functions
def process_frame(frame, model, conf, iou):
    """Process single frame with YOLOv8"""
    # Detect objects
    results = model(frame, conf=conf, iou=iou, verbose=False)
    
    # Draw boxes
    annotated_frame = results[0].plot()
    
    # Get detections info
    detections = defaultdict(int)
    boxes = results[0].boxes
    
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        detections[class_name] += 1
    
    return annotated_frame, detections, len(boxes)

def process_video(video_path, model, conf, iou):
    """Process video file"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    stframe = st.empty()
    col1, col2, col3 = st.columns(3)
    
    frame_count = 0
    total_detections = 0
    all_detections = defaultdict(int)
    
    progress_bar = st.progress(0)
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Process frame
        annotated_frame, detections, num_detections = process_frame(
            frame, model, conf, iou
        )
        
        # Update stats
        frame_count += 1
        total_detections += num_detections
        for class_name, count in detections.items():
            all_detections[class_name] += count
        
        # Convert BGR to RGB for display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        stframe.image(annotated_frame, use_column_width=True)
        
        # Update progress
        progress_bar.progress(frame_count / total_frames)
        
        # Calculate stats
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        with col1:
            st.metric("Frame", f"{frame_count}/{total_frames}")
        with col2:
            st.metric("FPS", f"{current_fps:.1f}")
        with col3:
            st.metric("Total Detections", total_detections)
    
    cap.release()
    
    return frame_count, total_detections, all_detections

# Process image
def process_image(image_path, model, conf, iou):
    """Process single image"""
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (640, 480))
    
    annotated_frame, detections, num_detections = process_frame(
        frame, model, conf, iou
    )
    
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame, detections, num_detections

# Main app logic
if input_source == "📷 Webcam":
    st.header("📷 Webcam Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        run_webcam = st.checkbox("📹 Start Webcam", value=False)
        frames_to_capture = st.number_input("Frames to capture", 1, 300, 30)
    
    if run_webcam:
        stframe = st.empty()
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        cap = cv2.VideoCapture(0)
        
        frame_count = 0
        total_detections = 0
        all_detections = defaultdict(int)
        start_time = time.time()
        
        while frame_count < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            
            # Process
            annotated_frame, detections, num_detections = process_frame(
                frame, model, conf_threshold, iou_threshold
            )
            
            frame_count += 1
            total_detections += num_detections
            for class_name, count in detections.items():
                all_detections[class_name] += count
            
            # Display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, use_column_width=True)
            
            # Stats
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            with stats_col1:
                st.metric("Frame", frame_count)
            with stats_col2:
                st.metric("FPS", f"{fps:.1f}")
            with stats_col3:
                st.metric("Objects Detected", total_detections)
        
        cap.release()
        
        # Summary statistics
        if all_detections:
            st.subheader("📊 Detection Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Objects Detected:**")
                for obj, count in sorted(all_detections.items(), 
                                        key=lambda x: x[1], reverse=True):
                    st.write(f"• {obj}: {count}")
            
            with col2:
                # Bar chart
                import pandas as pd
                df = pd.DataFrame(
                    list(all_detections.items()),
                    columns=['Object', 'Count']
                )
                st.bar_chart(df.set_index('Object'))

elif input_source == "📹 Upload Video":
    st.header("📹 Video Upload & Detection")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    if uploaded_video is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name
        
        st.info("⏳ Processing video... This may take a moment")
        
        frames_processed, total_dets, all_dets = process_video(
            video_path, model, conf_threshold, iou_threshold
        )
        
        # Summary
        st.success("✅ Video processing complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", frames_processed)
        with col2:
            st.metric("Total Detections", total_dets)
        with col3:
            st.metric("Avg Objects/Frame", 
                     f"{total_dets/frames_processed:.1f}" if frames_processed > 0 else "0")
        
        # Detection breakdown
        if all_dets:
            st.subheader("📊 Detection Breakdown")
            import pandas as pd
            df = pd.DataFrame(
                list(all_dets.items()),
                columns=['Object', 'Count']
            ).sort_values('Count', ascending=False)
            
            st.bar_chart(df.set_index('Object'))
            
            with st.expander("View detailed statistics"):
                st.write(df)
        
        # Cleanup
        os.remove(video_path)

else:  # Image upload
    st.header("🖼️ Image Detection")
    
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_image is not None:
        # Save and process
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_image.read())
            image_path = tmp.name
        
        annotated_img, detections, num_dets = process_image(
            image_path, model, conf_threshold, iou_threshold
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(annotated_img, use_column_width=True)
        
        with col2:
            st.metric("Objects Detected", num_dets)
            
            if detections:
                st.write("**Detected Objects:**")
                for obj, count in sorted(detections.items(), 
                                        key=lambda x: x[1], reverse=True):
                    st.write(f"• {obj}: {count}")
        
        # Cleanup
        os.remove(image_path)

# Footer
st.markdown("---")
st.markdown("""
    **Model Information:**
    - YOLOv8 Architecture
    - Trained on COCO Dataset (80 classes)
    - Real-time object detection
    
    **Need help?** Check the documentation or GitHub
""")
