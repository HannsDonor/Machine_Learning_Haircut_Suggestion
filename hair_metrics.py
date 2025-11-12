from fastapi import FastAPI, UploadFile, File
import os, cv2, numpy as np, math, tempfile, traceback
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAIR_MODEL_PATH = os.path.join(SCRIPT_DIR, "hair_segmenter.tflite")

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh

def _safe_div(a, b, eps=1e-6):
    return a / b if abs(b) > eps else 0.0

def init_segmenter():
    base_options = python.BaseOptions(model_asset_path=HAIR_MODEL_PATH)
    seg_options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    return vision.ImageSegmenter.create_from_options(seg_options)

segmenter = init_segmenter()

def analyze_hair(img):
    if img is None:
        return {"error": "Cannot read image."}

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as mesh:
        results = mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {"error": "No face detected."}
        pts = [(lm.x * w, lm.y * h) for lm in results.multi_face_landmarks[0].landmark]

    top_forehead_y = pts[10][1]
    chin_y = pts[152][1]
    head_height = max(chin_y - top_forehead_y, 1)

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        seg_result = segmenter.segment(mp_image)
        mask = seg_result.category_mask.numpy_view()
        hair_mask = (mask == 1).astype(np.uint8)
    except Exception:
        hair_mask = np.zeros((h, w), dtype=np.uint8)

    kernel = np.ones((5,5), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(hair_mask > 0)
    if len(ys) == 0:
        return {"error": "No hair detected."}

    top_y, bottom_y = np.min(ys), np.max(ys)
    left_x, right_x = np.min(xs), np.max(xs)
    hair_height = bottom_y - top_y
    hair_width = right_x - left_x

    hair_ratio = _safe_div(hair_height, head_height)
    vertical_extent = _safe_div((bottom_y - top_forehead_y), h)
    coverage_top = (np.sum(hair_mask[:max(1, int(h/4)), :]) / (w*max(1, int(h/4)))) * 100
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hair_pixels = np.sum(hair_mask)
    texture_estimate = float(np.sum(cv2.Canny(gray, 100, 200)[hair_mask>0]) / hair_pixels) if hair_pixels>0 else 0.0

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

    return {
        "hair_ratio": float(round(hair_ratio, 3)),
        "vertical_extent": float(round(vertical_extent, 3)),
        "hair_width": float(round(hair_width, 3)),
        "coverage_top_percent": float(round(coverage_top, 2)),
        "texture_estimate": float(round(texture_estimate, 3)),
        "gender": gender
    }
