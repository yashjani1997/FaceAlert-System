import cv2
import pickle
import winsound
import os
import numpy as np

# ========== CNN Face Detector (OpenCV DNN) ==========
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ========== Load trained LBPH model ==========
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print(
        "[ERROR] OpenCV was built without the 'face' module.\n"
        "Install the contrib package: pip install opencv-contrib-python"
    )
    exit(1)

if not os.path.exists("trainer.yml"):
    print("[ERROR] trainer.yml not found. Run the training script first (train.py).")
    exit(1)

recognizer.read("trainer.yml")

# ========== Load label mappings ==========
with open("labels.pickle", "rb") as f:
    label_ids = pickle.load(f)

id_to_name = {v: k for k, v in label_ids.items()}
print("[INFO] Loaded labels:", id_to_name)

# ========== Chor list ==========
CHORS = ["yash", "viren"]

# ========== Start webcam ==========
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    (h, w) = frame.shape[:2]

    # ---------- CNN Face Detection ----------
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            roi_gray = gray[y1:y2, x1:x2]
            roi_gray = cv2.resize(roi_gray, (200, 200))

            # ---------- LBPH Recognition ----------
            id_, conf = recognizer.predict(roi_gray)

            if conf < 85:
                name = id_to_name.get(id_, "Unknown")
            else:
                name = "Unknown"

            label_text = f"{name} ({int(conf)})"

            # ---------- Chor Alert ----------
            if name in CHORS:
                color = (0, 0, 255)
                winsound.Beep(2000, 300)
                cv2.putText(
                    frame,
                    f"⚠ CHOR DETECTED: {name}!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    cv2.imshow("Security Camera - Chor Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
