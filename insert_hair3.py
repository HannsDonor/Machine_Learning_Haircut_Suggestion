import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
from tkinter import simpledialog, Tk

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hair_segmenter.tflite")
CSV_PATH = os.path.join(SCRIPT_DIR, "haircut_v2.csv")

# === Mediapipe setup ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Hair segmentation model not found: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
segmenter = vision.ImageSegmenter.create_from_options(options)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# === CSV init ===
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["haircut_name", "face_shape", "gender",
                         "hair_ratio", "vertical_extent", "hair_width", "coverage_top", "texture_estimate"])

# === Tkinter setup ===
root = Tk()
root.withdraw()

# === Webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam.")

print("🎥 Webcam started.")
print("➡️ Press SPACE to edit details (haircut, face shape, gender).")
print("➡️ Press ENTER to capture 10 frames after entering details.")
print("➡️ Press ESC to exit.")

haircut_name = None
face_shape = None
gender = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask.numpy_view()
    category_mask = cv2.resize(category_mask, (w, h))
    hair_mask = (category_mask == 1).astype(np.uint8)

    overlay = frame.copy()
    overlay[hair_mask > 0] = (0, 255, 0)
    preview = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    cv2.putText(preview, "SPACE: Edit details", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(preview, "ENTER: Capture (10 frames)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(preview, "ESC: Quit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Display current info if set
    if haircut_name:
        cv2.putText(preview, f"Haircut: {haircut_name}", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, f"Face Shape: {face_shape}", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, f"Gender: {gender}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hair Capture", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == 32:  # SPACEBAR → edit details
        haircut_name = simpledialog.askstring("Haircut", "Enter haircut name:")
        face_shape = simpledialog.askstring("Face Shape", "Enter face shape:")
        gender = simpledialog.askstring("Gender", "Enter gender:")
        if haircut_name and face_shape and gender:
            print(f"✅ Details updated: {haircut_name}, {face_shape}, {gender}")
        else:
            print("⚠️ Missing details, please re-enter using SPACE.")

    elif key == 13:  # ENTER → capture 10 frames
        if not (haircut_name and face_shape and gender):
            print("⚠️ Please enter details first using SPACE.")
            continue

        print(f"🧠 Capturing 10 frames for {haircut_name} ({gender}, {face_shape})...")
        frame_data = []

        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask.numpy_view()
            category_mask = cv2.resize(category_mask, (w, h))
            hair_mask = (category_mask == 1).astype(np.uint8)

            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
                top_forehead_y = pts[10][1]
                chin_y = pts[152][1]
            else:
                top_forehead_y, chin_y = int(h * 0.3), int(h * 0.8)

            ys, xs = np.where(hair_mask > 0)
            if len(ys) == 0:
                continue

            top_y, bottom_y = np.min(ys), np.max(ys)
            left_x, right_x = np.min(xs), np.max(xs)
            hair_height = bottom_y - top_y
            hair_width = right_x - left_x

            # Facial scale stabilization
            scale_factor = 1.0
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left_eye = np.array([lm[33].x * w, lm[33].y * h])
                right_eye = np.array([lm[263].x * w, lm[263].y * h])
                eye_distance = np.linalg.norm(right_eye - left_eye)
                if eye_distance > 0:
                    scale_factor = 1.0 / (eye_distance / 100.0)

            head_height = chin_y - top_forehead_y
            hair_ratio = (hair_height / head_height * scale_factor) if head_height > 0 else 0
            vertical_extent = (bottom_y - top_forehead_y) / h
            coverage_top = np.sum(hair_mask[:int(h / 4), :]) / (w * (h / 4)) * 100
            hair_pixels = np.sum(hair_mask)
            texture_estimate = 0
            if hair_pixels > 0:
                edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
                texture_estimate = np.sum(edges[hair_mask > 0]) / hair_pixels

            live_preview = frame.copy()
            cv2.putText(live_preview, f"Frame {i+1}/10", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(live_preview, f"Hair Ratio: {hair_ratio:.3f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            overlay = live_preview.copy()
            overlay[hair_mask > 0] = (0, 255, 0)
            blended = cv2.addWeighted(live_preview, 0.6, overlay, 0.4, 0)
            cv2.imshow("Hair Capture", blended)
            cv2.waitKey(150)

            frame_data.append([haircut_name, face_shape, gender,
                               hair_ratio, vertical_extent, hair_width, coverage_top, texture_estimate])

        if frame_data:
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(frame_data)
            print(f"✅ Saved {len(frame_data)} frames for {haircut_name} ({gender}, {face_shape})")
        else:
            print("⚠️ No valid frames captured.")

cap.release()
cv2.destroyAllWindows()

# === Show total dataset count ===
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'r') as f:
        total = sum(1 for _ in f) - 1
        print(f"📊 Total records in haircut_v2.csv: {total}")
