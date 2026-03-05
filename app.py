import cv2
import pickle
import os
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="🔐 Chor Alert System", page_icon="🚨", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b0f1a; color: white; }
.alert-box {
    background-color: #ff000033;
    border: 2px solid red;
    border-radius: 10px;
    padding: 15px;
    font-size: 24px;
    font-weight: bold;
    color: red;
    text-align: center;
}
.safe-box {
    background-color: #00ff0022;
    border: 2px solid green;
    border-radius: 10px;
    padding: 15px;
    font-size: 20px;
    color: #00ff00;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔐 Face Recognition — Chor Alert System")
st.markdown("---")

# ========== CHOR LIST ==========
CHORS = ["yash", "viren"]

# ========== LOAD MODELS ==========
@st.cache_resource
def load_models():
    # CNN Face Detector
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    # LBPH Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    # Labels
    with open("labels.pickle", "rb") as f:
        label_ids = pickle.load(f)
    id_to_name = {v: k for k, v in label_ids.items()}

    return net, recognizer, id_to_name

try:
    net, recognizer, id_to_name = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Model load error: {e}")

# ========== RTC CONFIG (STUN server for WebRTC) ==========
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ========== SHARED STATE ==========
if "detected_name" not in st.session_state:
    st.session_state.detected_name = "No Face"
if "is_chor" not in st.session_state:
    st.session_state.is_chor = False

# ========== VIDEO PROCESSOR ==========
class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.detected_name = "No Face"
        self.is_chor = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        (h, w) = img.shape[:2]

        # CNN Detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        self.detected_name = "No Face"
        self.is_chor = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")

                # Safety clipping
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                roi_gray = gray[y1:y2, x1:x2]
                if roi_gray.size == 0:
                    continue
                roi_gray = cv2.resize(roi_gray, (200, 200))

                # LBPH Recognition
                id_, conf = recognizer.predict(roi_gray)
                name = id_to_name.get(id_, "Unknown") if conf < 85 else "Unknown"

                self.detected_name = name
                self.is_chor = name in CHORS

                # Draw box
                color = (0, 0, 255) if self.is_chor else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{name} ({int(conf)})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

                # Alert overlay on frame
                if self.is_chor:
                    cv2.putText(img, f"!! CHOR DETECTED: {name.upper()} !!",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 3)
                    # Red border overlay
                    cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 8)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========== UI LAYOUT ==========
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Camera Feed")
    if models_loaded:
        ctx = webrtc_streamer(
            key="chor-alert",
            video_processor_factory=FaceProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.warning("Models not loaded. Please check model files.")

with col2:
    st.markdown("### 🚨 Detection Status")
    status_placeholder = st.empty()
    
    st.markdown("---")
    st.markdown("### 👥 Monitored Persons")
    for chor in CHORS:
        st.error(f"🚫 {chor.upper()} — Intruder")

    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    st.markdown("""
    - **Detector:** CNN (SSD + ResNet-10)
    - **Recognizer:** LBPH
    - **Confidence threshold:** 85
    - **Detection confidence:** 0.6
    """)

# ========== AUTO REFRESH STATUS ==========
import time
if models_loaded and ctx.video_processor:
    processor = ctx.video_processor
    name = processor.detected_name
    is_chor = processor.is_chor

    with status_placeholder.container():
        if is_chor:
            st.markdown(f"""
            <div class='alert-box'>
                🚨 CHOR DETECTED!<br>{name.upper()}
            </div>
            """, unsafe_allow_html=True)
        elif name != "No Face":
            st.markdown(f"""
            <div class='safe-box'>
                ✅ Known Person<br>{name.upper()}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👁️ Monitoring... No face detected")