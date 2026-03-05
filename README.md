# 🔐 FaceAlert System — Real-time Intruder Detection

A real-time **Face Recognition Alert System** built using Computer Vision and Machine Learning. The system detects and recognizes faces via live camera feed and triggers instant visual alerts when a known intruder is identified.

---

## 🌐 Project Links

- 📄 **Project Page:** [yashjani1997.github.io/FaceAlert-System](https://yashjani1997.github.io/FaceAlert-System/)
- 💻 **GitHub Repo:** [yashjani1997/FaceAlert-System](https://github.com/yashjani1997/FaceAlert-System)
- 🎬 **Demo Video:** [Watch on YouTube](https://youtu.be/ll9HRZShkrc)

---

## 📌 Problem Statement

Traditional surveillance systems rely heavily on human monitoring — inefficient and prone to human error. Security personnel cannot continuously monitor multiple camera feeds effectively.

**FaceAlert System** solves this by automating the detection and identification process — triggering instant alerts when a known intruder is detected.

---

## 🏗️ Pipeline

```
Camera Input → CNN Face Detection → Face Crop → LBPH Recognition → Alert Trigger
```

| Step | Description |
|---|---|
| Camera Input | Live webcam / browser camera feed |
| CNN Detection | SSD + ResNet-10 detects faces in each frame |
| Face Crop | Detected face region extracted |
| LBPH Recognition | Predicts identity from grayscale face |
| Alert Trigger | Visual alert if intruder detected |

---

## 🚨 Alert System

| Status | Visual |
|---|---|
| **Intruder Detected** | Red bounding box + `!! CHOR DETECTED: NAME !!` overlay + Red border on full frame |
| **Known Person** | Green bounding box + name with confidence score |
| **Unknown** | Yellow bounding box — person not in training data |

---

## 🧠 Models

### 1. CNN Face Detector — OpenCV DNN (SSD + ResNet-10)

- **Model:** SSD with ResNet-10 backbone
- **Input Size:** 300 x 300 px
- **Confidence Threshold:** > 0.6
- **Files:** `deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`
- **Advantage:** More robust than Haar Cascade — works better under varying lighting and distances

### 2. LBPH Face Recognizer

- **Algorithm:** Local Binary Pattern Histogram (LBPH)
- **Input:** 200 x 200 Grayscale face crop
- **Recognition Threshold:** confidence < 85 = Known person
- **Training Output:** `trainer.yml` + `labels.pickle`
- **Training:** Custom dataset — per-person image folders

---

## 📂 Project Structure

```
FaceAlert-System/
├── dataset/              # Training images (per-person folders)
│   ├── person1/
│   └── person2/
├── deploy.prototxt       # CNN detector config
├── labels.pickle         # Label ID → Name mapping
├── trainer.yml           # Trained LBPH model
├── train.py              # Training script
├── recognize.py          # Original OpenCV recognition script
├── app.py                # Streamlit web app (streamlit-webrtc)
├── requirements.txt      # Dependencies
└── index.html            # Project info page
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| OpenCV DNN | CNN-based face detection |
| OpenCV Face (LBPH) | Face recognition |
| Streamlit | Web dashboard |
| streamlit-webrtc | Live browser camera feed |
| NumPy | Image processing |

---

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yashjani1997/FaceAlert-System.git
cd FaceAlert-System

# Install dependencies
pip install -r requirements.txt

# Add training images
# dataset/yourname/img1.jpg, img2.jpg ...

# Train the model
python train.py

# Run the app
streamlit run app.py
```

---

## 🗂️ Dataset Preparation

```
dataset/
├── yash/
│   ├── img1.jpg
│   └── img2.jpg
├── viren/
│   ├── img1.jpg
│   └── img2.jpg
```

- Minimum **20-30 images** per person recommended
- Different angles and lighting conditions improve accuracy

---

## ⚠️ Limitations

- Recognition accuracy depends on training dataset quality
- Extreme lighting conditions may affect detection
- Requires clear face visibility
- LBPH performance decreases with very large number of classes

---

## 🔮 Future Improvements

- Replace LBPH with **ArcFace / FaceNet** for higher accuracy
- Send alerts to **mobile devices or email**
- **Auto-save** intruder images with timestamp
- Integrate with **CCTV / IP cameras**

---

## 🧠 Key Learnings

- CNN-based face detection vs traditional Haar Cascade
- LBPH face recognition pipeline
- Streamlit + streamlit-webrtc for live browser camera
- End-to-end computer vision security system design

---

## 👤 Author

**Yash Jani**  
Data Analyst & Machine Learning Enthusiast  
[GitHub: yashjani1997](https://github.com/yashjani1997)
