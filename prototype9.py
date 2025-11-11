import os
import math
import traceback
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAIR_MODEL_PATH = os.path.join(SCRIPT_DIR, "hair_segmenter.tflite")
FACE_DATASET = os.path.join(SCRIPT_DIR, "face_shape_dataset.csv")
HAIRCUT_DATASET = os.path.join(SCRIPT_DIR, "haircut_v2.csv")

_initialized = False
_scaler: Optional[StandardScaler] = None
_rf_face: Optional[RandomForestClassifier] = None
_le: Optional[LabelEncoder] = None
_segmenter: Optional[vision.ImageSegmenter] = None
_mp_face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
_haircut_df: Optional[pd.DataFrame] = None

_FEATURES = [
    "face_length", "forehead_width", "cheek_width", "jaw_width",
    "face_ratio", "jaw_to_forehead", "cheek_to_forehead",
    "left_angle", "right_angle"
]

def init_models() -> None:
    global _initialized, _scaler, _rf_face, _le, _segmenter, _mp_face_mesh, _haircut_df
    if _initialized:
        print("prototype9: already initialized", flush=True)
        return
    try:
        if not os.path.exists(FACE_DATASET):
            raise FileNotFoundError(f"Missing dataset: {FACE_DATASET}")
        if not os.path.exists(HAIR_MODEL_PATH):
            raise FileNotFoundError(f"Missing model file: {HAIR_MODEL_PATH}")

        face_df = pd.read_csv(FACE_DATASET).dropna(subset=_FEATURES + ["label"]).reset_index(drop=True)
        X = face_df[_FEATURES].astype(float)
        y = face_df["label"].astype(str)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf_face = RandomForestClassifier(n_estimators=150, random_state=42)
        rf_face.fit(X_scaled, y_enc)
        _scaler = scaler
        _rf_face = rf_face
        _le = le

        base_options = python.BaseOptions(model_asset_path=HAIR_MODEL_PATH)
        seg_options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        _segmenter = vision.ImageSegmenter.create_from_options(seg_options)

        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, refine_landmarks=True, max_num_faces=1
        )

        if os.path.exists(HAIRCUT_DATASET):
            df = pd.read_csv(HAIRCUT_DATASET)
            df["face_shape"] = df["face_shape"].astype(str).str.strip().str.upper()
            df["gender"] = df["gender"].astype(str).str.strip().str.lower().replace({
                "malw": "male", "femal": "female", "m": "male", "f": "female"
            })
            df["haircut_name"] = df["haircut_name"].astype(str).str.strip().str.title()
            _haircut_df = df
        else:
            _haircut_df = None

        _initialized = True
        print("prototype9: init_models completed", flush=True)
    except Exception:
        print("prototype9: init_models FAILED", flush=True)
        traceback.print_exc()
        raise

def _ensure_initialized() -> None:
    if not _initialized:
        raise RuntimeError("prototype9 not initialized. Call init_models() first.")

def _euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _calculate_angle(a, b, c) -> float:
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return ang + 360 if ang < 0 else ang

def _safe_div(a, b, eps=1e-6) -> float:
    return a / b if abs(b) > eps else 0.0

def analyze_frame(frame_bgr: np.ndarray) -> Dict[str, Any]:
    _ensure_initialized()
    scaler = _scaler
    rf_face = _rf_face
    le = _le
    segmenter = _segmenter
    face_mesh = _mp_face_mesh
    haircut_df = _haircut_df

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    # Hair segmentation
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        seg_result = segmenter.segment(mp_image)
        mask = seg_result.category_mask.numpy_view()
        hair_mask = (mask == 1).astype(np.uint8)
    except Exception:
        hair_mask = np.zeros((h, w), dtype=np.uint8)

    # Clean mask
    try:
        kernel = np.ones((5, 5), np.uint8)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
    except Exception:
        pass

    # Face landmarks and shape prediction — EXACTLY as in standalone debug script
    try:
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {"error": "No face detected"}

        points = [(lm.x * w, lm.y * h) for lm in results.multi_face_landmarks[0].landmark]

        # Reference: inter-eye distance (points 33 and 263)
        ref_len = _euclidean(points[33], points[263])
        if ref_len < 5:
            ref_len = max(ref_len, 1.0)  # fallback to avoid zero division

        # Key landmarks — indices identical to debug script
        top_forehead = np.array(points[10])
        chin = np.array(points[152])
        left_temple = np.array(points[71])
        right_temple = np.array(points[301])
        left_cheek = np.array(points[227])
        right_cheek = np.array(points[447])
        left_jaw = np.array(points[172])
        right_jaw = np.array(points[435])

        # Compute features — formulas identical
        face_length = _euclidean(top_forehead, chin) / ref_len
        forehead_width = _euclidean(left_temple, right_temple) / ref_len
        cheek_width = _euclidean(left_cheek, right_cheek) / ref_len
        jaw_width = _euclidean(left_jaw, right_jaw) / ref_len

        face_ratio = _safe_div(face_length, cheek_width)
        jaw_to_forehead = _safe_div(jaw_width, forehead_width)
        cheek_to_forehead = _safe_div(cheek_width, forehead_width)
        left_angle = _calculate_angle(left_cheek, left_temple, chin)
        right_angle = -_calculate_angle(right_cheek, right_temple, chin)

        # Prediction — identical pipeline
        feat = np.array([[face_length, forehead_width, cheek_width, jaw_width,
                          face_ratio, jaw_to_forehead, cheek_to_forehead,
                          left_angle, right_angle]])
        feat_scaled = scaler.transform(feat)
        pred_enc = rf_face.predict(feat_scaled)[0]
        pred_shape = le.inverse_transform([pred_enc])[0].strip().upper()

    except Exception:
        traceback.print_exc()
        pred_shape = "UNKNOWN"

    # Hair metrics
    try:
        ys, xs = np.where(hair_mask > 0)
        if len(ys) > 0:
            top_y, bottom_y = np.min(ys), np.max(ys)
            left_x, right_x = np.min(xs), np.max(xs)
            hair_height = bottom_y - top_y
            hair_width = right_x - left_x
            top_forehead_y = int(points[10][1])
            chin_y = int(points[152][1])
            head_height = chin_y - top_forehead_y if chin_y > top_forehead_y else max(1, int(h * 0.5))
            hair_ratio = _safe_div(hair_height, head_height)
            vertical_extent = _safe_div((bottom_y - top_forehead_y), h)
            coverage_top = (np.sum(hair_mask[:max(1, int(h / 4)), :]) / (w * max(1, int(h / 4)))) * 100
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            hair_pixels = np.sum(hair_mask)
            texture_estimate = 0.0
            if hair_pixels > 0:
                edges = cv2.Canny(gray, 100, 200)
                texture_estimate = float(np.sum(edges[hair_mask > 0]) / hair_pixels)
        else:
            hair_ratio = vertical_extent = coverage_top = texture_estimate = hair_width = 0.0
    except Exception:
        traceback.print_exc()
        hair_ratio = vertical_extent = coverage_top = texture_estimate = hair_width = 0.0

    # Gender detection via DeepFace
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        analysis = DeepFace.analyze(
            img_path=temp_path,
            actions=["gender"],
            enforce_detection=False,
            detector_backend="retinaface",
        )

        os.remove(temp_path)

        detected = analysis[0]["gender"] if isinstance(analysis, list) else analysis.get("gender")
        if isinstance(detected, dict):
            gender = "Male" if detected.get("Man", 0) > detected.get("Woman", 0) else "Female"
        else:
            gender = str(detected).capitalize()
    except Exception:
        traceback.print_exc()
        gender = "Unknown"

    results_dict: Dict[str, Any] = {
        "face_shape": pred_shape,
        "gender": gender,
        "hair_ratio": round(float(hair_ratio), 3),
        "vertical_extent": round(float(vertical_extent), 3),
        "hair_width": round(float(hair_width), 3),
        "coverage_top": round(float(coverage_top), 2),
        "texture_estimate": round(float(texture_estimate), 3),
        "suggestions": []
    }

    # Haircut suggestions
    try:
        if haircut_df is not None:
            df = haircut_df
            gender_lower = gender.strip().lower()
            face_shape_upper = pred_shape.strip().upper()
            global_avg = df.groupby("haircut_name").agg({
                "hair_ratio": "mean",
                "vertical_extent": "mean",
                "hair_width": "mean",
                "coverage_top": "mean",
                "texture_estimate": "mean"
            }).reset_index()
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
                    suggestions = []
                    for row in merged.itertuples(index=False):
                        suggestions.append({
                            "name": row.haircut_name,
                            "count": int(row.count),
                            "avg_ratio": round(float(row.hair_ratio), 3)
                        })
                    results_dict["suggestions"] = suggestions
    except Exception:
        traceback.print_exc()

    return results_dict