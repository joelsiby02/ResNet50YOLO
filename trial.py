import streamlit as st
import os
import shutil

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
import cv2
import numpy as np
import glob
import time
from ultralytics import YOLO
from streamlit_extras.let_it_rain import rain
import atexit

# ---------------------- Constants & Config ----------------------
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
ALLOWED_VIDEO_TYPES = ["mp4", "avi", "mov"]
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "runs/detect/predict"
MODEL_PATH = "best.pt"

# ---------------------- Model Loading ----------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------------- Helper Functions ----------------------
def cleanup():
    """Delete temp files when app closes."""
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree("runs", ignore_errors=True)

def success_animation():
    """Trigger a success animation."""
    rain(emoji="🎉", font_size=20, falling_speed=3, animation_length=1)

def get_latest_file(folder, extensions):
    """Fetch the latest file in a directory with specified extensions."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
    return max(files, key=os.path.getctime) if files else None

# ---------------------- Streamlit UI ----------------------
st.set_page_config(
    page_title="ISL Detector",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background: #f8f9fa; }
    .stButton>button { background: #4a90e2; color: white; border-radius: 8px; }
    </style>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🖼️ ISL Detection System")
    app_mode = st.radio("Select Mode", ["📷 Image Detection", "🎥 Video Analysis", "📸 Live Capture"])
    st.markdown("---")
    with st.expander("ℹ️ Instructions"):
        st.markdown("1. Select mode\n2. Upload media\n3. Detect objects")

# ---------------------- Image Detection ----------------------
if "Image" in app_mode:
    uploaded_file = st.file_uploader("Upload an image", type=ALLOWED_IMAGE_TYPES)
    
    if uploaded_file and st.button("🔍 Detect"):
        with st.spinner("Analyzing..."):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            # Save image
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run YOLO detection
            results = model.predict(image_path, save=True, project="runs/detect", name="predict", exist_ok=True)

            # Fetch processed image
            latest_image_path = get_latest_file(RESULT_FOLDER, ["jpg", "png"])

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_path, caption="Original Image", use_column_width=True)
            with col2:
                if latest_image_path:
                    st.image(latest_image_path, caption="Processed Image", use_column_width=True)
                    with open(latest_image_path, "rb") as file:
                        st.download_button(label="📥 Download Processed Image", data=file, file_name="result.jpg", mime="image/jpeg")
                    success_animation()
                else:
                    st.error("No processed image found!")

# ---------------------- Video Analysis ----------------------
elif "Video" in app_mode:
    uploaded_video = st.file_uploader("Upload a video", type=ALLOWED_VIDEO_TYPES)
    
    if uploaded_video and st.button("🔍 Analyze"):
        with st.spinner("Processing..."):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            # Save uploaded video
            video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            # Run YOLO detection
            results = model.predict(source=video_path, save=True, project="runs/detect", name="predict", exist_ok=True)

            # Wait for processing
            time.sleep(2)

            # Fetch processed video
            latest_video_path = get_latest_file(RESULT_FOLDER, ["mp4", "avi"])
            
            if latest_video_path:
                st.video(latest_video_path)
                with open(latest_video_path, "rb") as file:
                    st.download_button(label="📥 Download Processed Video", data=file, file_name="result.mp4", mime="video/mp4")
                st.success("Detection Completed!")
                success_animation()
            else:
                st.error("No processed video found!")

# ---------------------- Live Camera Mode ----------------------
else:
    st.write("Real-Time Sign Language Detection")

    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Live Detection 🚀") and not st.session_state.running:
            st.session_state.running = True
    with col2:
        if st.button("Stop Detection ⏹️") and st.session_state.running:
            st.session_state.running = False

    FRAME_WINDOW = st.empty()

    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while st.session_state.running:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame")
            st.session_state.running = False
            break

        # Process frame with YOLO model
        results = model.predict(frame)
        annotated_frame = results[0].plot()

        # Convert BGR to RGB and display
        FRAME_WINDOW.image(annotated_frame[..., ::-1], channels="RGB")

    if not st.session_state.running and 'cap' in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap
        st.rerun()

# Cleanup when app closes
atexit.register(cleanup)









