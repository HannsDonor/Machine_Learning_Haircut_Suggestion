import cv2
import numpy as np
import os
import math
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deepface import DeepFace

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DATASET = os.path.join(SCRIPT_DIR, "face_shape_dataset.csv")
HAIRCUT_DATASET = os.path.join(SCRIPT_DIR, "haircut_v2.csv")

# === Load Dataset & Train Face Shape Model ===
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

# === Helper Functions ===
def safe_div(a, b, eps=1e-6):
    return a / b if abs(b) > eps else 0

def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def calculate_angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    )
    return ang + 360 if ang < 0 else ang

# === Start Webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access webcam.")

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

print("🎥 Webcam started — Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.multi_face_landmarks:
        points = [(lm.x*w, lm.y*h) for lm in results.multi_face_landmarks[0].landmark]
        top_forehead = np.array(points[10])
        chin = np.array(points[152])
        left_temple = np.array(points[71])
        right_temple = np.array(points[301])
        left_cheek = np.array(points[227])
        right_cheek = np.array(points[447])
        left_jaw = np.array(points[172])
        right_jaw = np.array(points[435])

        face_length = euclidean(top_forehead, chin)
        forehead_width = euclidean(left_temple, right_temple)
        cheek_width = euclidean(left_cheek, right_cheek)
        jaw_width = euclidean(left_jaw, right_jaw)

        face_ratio = safe_div(face_length, cheek_width)
        jaw_to_forehead = safe_div(jaw_width, forehead_width)
        cheek_to_forehead = safe_div(cheek_width, forehead_width)
        left_angle = calculate_angle(left_cheek, left_temple, chin)
        right_angle = -calculate_angle(right_cheek, right_temple, chin)

        feat = np.array([[face_length, forehead_width, cheek_width, jaw_width,
                          face_ratio, jaw_to_forehead, cheek_to_forehead,
                          left_angle, right_angle]])
        feat_scaled = scaler.transform(feat)
        pred_enc = rf_face.predict(feat_scaled)[0]
        pred_shape = le.inverse_transform([pred_enc])[0].strip().upper()

        # Map short code to full word
        shape_map = {
            "O": "Oblong",
            "R": "Round",
            "H": "Heart",
            "D": "Diamond",
            "S": "Square"
        }
        shape_name = shape_map.get(pred_shape[:1], pred_shape)

        # Draw landmarks (optional small dots)
        for (x, y) in points[::15]:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Display face shape
        cv2.putText(frame, f"Face Shape: {shape_name}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Live Face Shape Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
