import os
import cv2
import numpy as np
import pickle

DATASET_DIR = "dataset"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = []
labels = []
label_ids = {}
current_id = 0

print("[INFO] Loading images from dataset...")
print(f"[INFO] Dataset path: {os.path.abspath(DATASET_DIR)}")

for root, dirs, files in os.walk(DATASET_DIR):
    person_name = os.path.basename(root)

    # skip root if it is exactly "dataset"
    if person_name == "dataset":
        continue

    if not files:
        print(f"[WARN] No files in folder: {root}")
        continue

    if person_name not in label_ids:
        label_ids[person_name] = current_id
        current_id += 1

    label_id = label_ids[person_name]

    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".jfif")):
            print(f"[WARN] Skipping non-image file: {file}")
            continue

        image_path = os.path.join(root, file)
        print(f"[INFO] Processing: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_rect = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,       # thoda loose rakha
            minSize=(40, 40)      # chhota face bhi detect ho sake
        )

        if len(faces_rect) == 0:
            print(f"[WARN] No face detected in: {image_path}")
            continue

        for (x, y, w, h) in faces_rect:
            roi_gray = gray[y:y + h, x:x + w]
            faces.append(roi_gray)
            labels.append(label_id)

print(f"[INFO] Total faces found: {len(faces)}")
print(f"[INFO] Label IDs: {label_ids}")

if len(faces) == 0:
    print("[ERROR] No faces found in dataset. Please use clear face photos.")
    exit()

print("[INFO] Training model...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

recognizer.write("trainer.yml")
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

print("[INFO] Training complete ✅")
print("[INFO] Model saved as trainer.yml and labels.pickle")
