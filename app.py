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
    if not models_ready or _analyze_frame is None:
        return JSONResponse(status_code=503, content={"error": "Model not ready"})

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        result = _analyze_frame(img)
        return JSONResponse(content={"suggestion": result})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
