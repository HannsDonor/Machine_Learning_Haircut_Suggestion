import cv2
import numpy as np
import os
import math
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DATASET = os.path.join(SCRIPT_DIR, "face_shape_dataset.csv")
HAIRCUT_DATASET = os.path.join(SCRIPT_DIR, "haircut_v2.csv")

# === Load Datasets ===
if not os.path.exists(FACE_DATASET):
    raise FileNotFoundError("❌ Missing face_shape_dataset.csv")

face_df = pd.read_csv(FACE_DATASET)
features = [
    "face_length", "forehead_width", "cheek_width", "jaw_width",
    "face_ratio", "jaw_to_forehead", "cheek_to_forehead",
    "left_angle", "right_angle"
]

face_df = face_df.dropna(subset=features + ["label"]).reset_index(drop=True)
X = face_df[features].astype(float)
y = face_df["label"].astype(str)

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_face = RandomForestClassifier(n_estimators=150, random_state=42)
rf_face.fit(X_scaled, y_enc)

# === Choose image ===
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Select image to analyze",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

if not image_path:
    print("❌ No image selected.")
    exit()

# === Load Image ===
frame = cv2.imread(image_path)
if frame is None:
    print("❌ Cannot read image.")
    exit()

h, w, _ = frame.shape
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# === Hair Segmentation (classic MediaPipe SelfieSegmentation) ===
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)  # 0: general, 1: landscape/portrait

results = segmenter.process(rgb)
mask = results.segmentation_mask  # float mask 0..1
hair_mask = (mask > 0.5).astype(np.uint8)  # binary mask

# Clean mask
kernel = np.ones((5,5), np.uint8)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)

# === Face Mesh for feature extraction ===
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
results_face = mp_face.process(rgb)

if not results_face.multi_face_landmarks:
    print("⚠️ No face detected.")
    exit()

points = [(lm.x*w, lm.y*h) for lm in results_face.multi_face_landmarks[0].landmark]
top_forehead = np.array(points[10])
chin = np.array(points[152])
left_temple = np.array(points[71])
right_temple = np.array(points[301])
left_cheek = np.array(points[227])
right_cheek = np.array(points[447])
left_jaw = np.array(points[172])
right_jaw = np.array(points[435])

def safe_div(a, b, eps=1e-6):
    return a / b if abs(b) > eps else 0

def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def calculate_angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    )
    return ang + 360 if ang < 0 else ang

face_length = euclidean(top_forehead, chin)
forehead_width = euclidean(left_temple, right_temple)
cheek_width = euclidean(left_cheek, right_cheek)
jaw_width = euclidean(left_jaw, right_jaw)

face_ratio = safe_div(face_length, cheek_width)
jaw_to_forehead = safe_div(jaw_width, forehead_width)
cheek_to_forehead = safe_div(cheek_width, forehead_width)
left_angle = calculate_angle(left_cheek, left_temple, chin)
right_angle = -calculate_angle(right_cheek, right_temple, chin)

# === Predict Face Shape ===
feat = np.array([[face_length, forehead_width, cheek_width, jaw_width,
                  face_ratio, jaw_to_forehead, cheek_to_forehead,
                  left_angle, right_angle]])
feat_scaled = scaler.transform(feat)
pred_enc = rf_face.predict(feat_scaled)[0]
pred_shape = le.inverse_transform([pred_enc])[0].strip().upper()

# === Hair metrics ===
ys, xs = np.where(hair_mask > 0)
if len(ys) > 0:
    top_y, bottom_y = np.min(ys), np.max(ys)
    left_x, right_x = np.min(xs), np.max(xs)
    hair_height = bottom_y - top_y
    hair_width = right_x - left_x

    top_forehead_y = int(points[10][1])
    chin_y = int(points[152][1])
    head_height = chin_y - top_forehead_y if chin_y > top_forehead_y else h * 0.5

    hair_ratio = safe_div(hair_height, head_height)
    vertical_extent = safe_div((bottom_y - top_forehead_y), h)
    coverage_top = np.sum(hair_mask[:int(h/4), :]) / (w*(h/4)) * 100

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hair_pixels = np.sum(hair_mask)
    texture_estimate = 0
    if hair_pixels > 0:
        edges = cv2.Canny(gray, 100, 200)
        texture_estimate = np.sum(edges[hair_mask > 0]) / hair_pixels
else:
    hair_ratio = vertical_extent = coverage_top = texture_estimate = hair_width = 0

# === Gender Detection ===
try:
    analysis = DeepFace.analyze(img_path=image_path, actions=["gender"], enforce_detection=False)
    detected = analysis[0]["gender"]
    if isinstance(detected, dict):
        gender = "Male" if detected.get("Man", 0) > detected.get("Woman", 0) else "Female"
    else:
        gender = str(detected).capitalize()
except Exception:
    gender = "Unknown"

# === Display results ===
print("\n===== ANALYSIS RESULTS =====")
print(f"Face Shape: {pred_shape}")
print(f"Gender: {gender}")
print(f"Hair Ratio: {hair_ratio:.3f}")
print(f"Vertical Extent: {vertical_extent:.3f}")
print(f"Hair Width: {hair_width:.3f}")
print(f"Coverage Top: {coverage_top:.2f}%")
print(f"Texture Estimate: {texture_estimate:.3f}")
print("============================")

# === Haircut Suggestion ===
if os.path.exists(HAIRCUT_DATASET):
    df = pd.read_csv(HAIRCUT_DATASET)

    # Normalize formatting
    df["face_shape"] = df["face_shape"].astype(str).str.strip().str.upper()
    df["gender"] = df["gender"].astype(str).str.strip().str.lower().replace({
        "malw": "male", "femal": "female", "m": "male", "f": "female"
    })
    df["haircut_name"] = df["haircut_name"].astype(str).str.strip().str.title()

    gender_lower = gender.strip().lower()
    face_shape_upper = pred_shape.strip().upper()

    print(f"\n[DEBUG] Predicted shape: {face_shape_upper}, Gender: {gender_lower}")

    # Global averages
    global_avg = df.groupby("haircut_name").agg({
        "hair_ratio": "mean",
        "vertical_extent": "mean",
        "hair_width": "mean",
        "coverage_top": "mean",
        "texture_estimate": "mean"
    }).reset_index()

    # Local popularity
    subset = df[(df["face_shape"] == face_shape_upper) & (df["gender"] == gender_lower)]

    if not subset.empty:
        popularity = subset["haircut_name"].value_counts().reset_index()
        popularity.columns = ["haircut_name", "count"]

        merged = pd.merge(global_avg, popularity, on="haircut_name", how="inner")

        tolerance = 0.02
        merged = merged[merged["hair_ratio"] <= (hair_ratio + tolerance)]

        if not merged.empty:
            merged["ratio_diff"] = abs(merged["hair_ratio"] - hair_ratio)
            merged = merged.sort_values(by=["count", "ratio_diff"], ascending=[False, True])

            print("\n💇 Recommended Haircuts:")
            for i, row in enumerate(merged.itertuples(index=False), 1):
                print(f"{i}. {row.haircut_name} — {row.count} samples | avg ratio: {row.hair_ratio:.3f}")

            top_haircuts = merged["haircut_name"].tolist()
        else:
            print("\n⚠️ No shorter/similar haircuts found within this ratio range.")
            top_haircuts = []
    else:
        print(f"\n⚠️ No matching face shape/gender subset found for {face_shape_upper} / {gender_lower}.")
        top_haircuts = []
else:
    print("\n❌ haircut_dataset.csv not found.")
    top_haircuts = []
