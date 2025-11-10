from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import asyncio
import traceback
import sys
import cv2
import numpy as np
import prototype9  # Your model logic lives here

app = FastAPI()

models_ready = False
_analyze_frame = None

@app.on_event("startup")
async def startup_event():
    global models_ready, _analyze_frame
    try:
        print("Model loader: initializing real models", flush=True)
        await asyncio.to_thread(prototype9.init_models)
        _analyze_frame = prototype9.analyze_frame
        print("Model loader: finished", flush=True)
    except Exception:
        print("Model loader exception:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        models_ready = True
        print(f"Model loader: models_ready={models_ready}", flush=True)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"models_ready": models_ready}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}", flush=True)

    if not models_ready or _analyze_frame is None:
        print("Model not ready", flush=True)
        return JSONResponse(status_code=503, content={"error": "Model not ready"})

    try:
        contents = await file.read()
        print(f"File size: {len(contents)} bytes", flush=True)

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("Failed to decode image", flush=True)
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        print(f"Image shape: {img.shape}", flush=True)

        result = _analyze_frame(img)
        print(f"Prediction result: {result}", flush=True)

        return JSONResponse(content=result)
    except Exception as e:
        print("Exception in /predict:", flush=True)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
