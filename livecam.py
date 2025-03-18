import os
# For deployment in streamlit cloud
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Forces OpenCV to use non-GUI mode


import streamlit as st
import requests
import atexit
import shutil
import cv2
import threading
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from ultralytics import YOLO
from streamlit_extras.let_it_rain import rain


# ---------------------- Flask Backend Setup ----------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "runs/detect/predict"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read and process the image in memory
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO prediction
    results = model.predict(frame)
    annotated_frame = results[0].plot()  # Get annotated frame

    # Encode to JPEG and return
    _, img_encoded = cv2.imencode(".jpg", annotated_frame)
    return Response(img_encoded.tobytes(), mimetype="image/jpeg")

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# Start Flask in background thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# ---------------------- Streamlit Frontend ----------------------
# Configuration and cleanup
def cleanup():
    shutil.rmtree("uploads", ignore_errors=True)
    shutil.rmtree("runs", ignore_errors=True)
atexit.register(cleanup)

st.set_page_config(
    page_title="ISL Detector",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UI Styling
st.markdown("""
    <style>
    .main { background: #f8f9fa; }
    .stButton>button { background: #4a90e2; color: white; border-radius: 8px; }
    </style>""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🖼️ ISL Detection System")
    app_mode = st.radio("Select Mode", ["📷 Image Detection", "🎥 Video Analysis", "📸 Live Capture"])
    st.markdown("---")
    with st.expander("ℹ️ Instructions"):
        st.markdown("1. Select mode\n2. Upload media\n3. Detect objects")

# Main Application Logic
def handle_camera():
    st.write("Real-Time Sign Language Detection")
    run_live = st.button("Start Live Detection 🚀")
    FRAME_WINDOW = st.image([])

    if run_live:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Convert frame to JPEG
            _, img_buffer = cv2.imencode(".jpg", frame)
            img_bytes = img_buffer.tobytes()

            # Get prediction from Flask API
            response = requests.post(
                "http://localhost:5000/predict_frame",
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")}
            )

            if response.status_code == 200:
                # Display processed frame
                img_array = np.frombuffer(response.content, np.uint8)
                pred_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                FRAME_WINDOW.image(pred_frame, channels="BGR")
            else:
                st.error("Detection failed: " + response.text)
                break

        cap.release()

# Mode Routing
st.title("ISL - Real-Time Detection System")
if "Image" in app_mode:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file and st.button("🔍 Detect"):
        response = requests.post("http://localhost:5000/predict", files={"file": uploaded_file})
        if response.status_code == 200:
            col1, col2 = st.columns(2)
            with col1: st.image(uploaded_file, use_container_width=True)
            with col2: st.image(f"http://localhost:5000/results/{uploaded_file.name}", use_container_width=True)

elif "Video" in app_mode:
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi"])
    if uploaded_video and st.button("🔍 Analyze"):
        response = requests.post("http://localhost:5000/predict_video", files={"file": uploaded_video})
        if response.status_code == 200:
            result = response.json()
            st.video(f"http://localhost:5000/results/{result['result_video'].split('/')[-1]}")

else:
    handle_camera()

# Footer
st.markdown("---")
st.markdown("🚀 Powered by YOLOv8 | 🔐 Secure Processing", unsafe_allow_html=True)
