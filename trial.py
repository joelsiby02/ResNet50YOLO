import streamlit as st

# Move set_page_config to the top
st.set_page_config(
    page_title="ISL Detector",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import shutil
import cv2
import numpy as np
import glob
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
def convert_video_to_mp4(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, 
                         cap.get(cv2.CAP_PROP_FPS), 
                         (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
    cap.release()
    out.release()
    os.remove(input_path)

def cleanup():
    shutil.rmtree("uploads", ignore_errors=True)
    shutil.rmtree("runs", ignore_errors=True)

# ---------------------- Streamlit App ----------------------
# st.set_page_config(
#     page_title="ISL Detector",
#     page_icon="🖼️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

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

def success_animation():
    rain(emoji="🎉", font_size=20, falling_speed=3, animation_length=1)

# Main Application Logic
st.title("ISL - Real-Time Detection System")

# ---------------------- Image Detection ----------------------
if "Image" in app_mode:
    uploaded_file = st.file_uploader("Upload image", type=ALLOWED_IMAGE_TYPES)
    if uploaded_file and st.button("🔍 Detect"):
        with st.spinner("Analyzing..."):
            # Process image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            results = model.predict(image, save=True, project="runs/detect", name="predict", exist_ok=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, channels="BGR", use_container_width=True)
            with col2:
                annotated_image = results[0].plot()
                st.image(annotated_image[..., ::-1], use_container_width=True)  # Convert BGR to RGB
                success_animation()

# ---------------------- Video Analysis ----------------------
elif "Video" in app_mode:
    uploaded_video = st.file_uploader("Upload video", type=ALLOWED_VIDEO_TYPES)
    if uploaded_video and st.button("🔍 Analyze"):
        with st.spinner("Processing..."):
            # Save video
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            # Process video
            detected_classes = set()
            results = model.predict(source=video_path, save=True, project="runs/detect", name="predict", exist_ok=True)
            
            for result in results:
                for box in result.boxes:
                    detected_classes.add(model.names[int(box.cls.item())])
            
            # Convert output video
            base_name = os.path.splitext(uploaded_video.name)[0]
            avi_path = max(glob.glob(os.path.join(RESULT_FOLDER, f"{base_name}*.avi")), key=os.path.getctime)
            mp4_path = avi_path.replace(".avi", ".mp4")
            convert_video_to_mp4(avi_path, mp4_path)
            
            # Display results
            st.video(mp4_path)
            if detected_classes:
                st.subheader("Detected Signs:")
                st.write(", ".join(detected_classes))

# ---------------------- Live Camera Mode ----------------------
# ---------------------- Live Camera Mode ----------------------
else:
    st.write("Real-Time Sign Language Detection")
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    # Start/Stop controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Live Detection 🚀") and not st.session_state.running:
            st.session_state.running = True
    with col2:
        if st.button("Stop Detection ⏹️") and st.session_state.running:
            st.session_state.running = False
            
    FRAME_WINDOW = st.image([])
    
    # Camera initialization
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Frame processing loop
    if st.session_state.running:
        while st.session_state.running:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.error("Failed to capture frame")
                st.session_state.running = False
                break
            
            # Process frame
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            
            # Convert to RGB and display
            FRAME_WINDOW.image(annotated_frame[..., ::-1], channels="RGB")
            
        # Release resources when stopped
        st.session_state.cap.release()
        del st.session_state.cap
        st.rerun()

st.markdown("---")
st.markdown("🚀 Powered by YOLOv8 | 🔐 Secure Processing", unsafe_allow_html=True)

# Cleanup when app closes
atexit.register(cleanup)


