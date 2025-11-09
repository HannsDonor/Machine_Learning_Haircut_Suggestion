from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
import cv2
import sys

app = FastAPI()

# Allow quick browser testing (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Quick health check that returns immediately
@app.get("/health")
def health():
    return {"status": "ok"}

# Runtime flags/placeholders
models_ready = False
_analyze_frame = None

# Background loader to import heavy modules and initialize models
async def load_models():
    global models_ready, _analyze_frame
    try:
        # Import heavy code here to avoid blocking module import/startup
        from prototype9 import analyze_frame as af  # local import
        _analyze_frame = af

        # If prototype9 exposes an explicit init or weight loading call, call it here:
        # try:
        #     await maybe_async_init_in_prototype9()
        # except Exception as e:
        #     print("Model init failed:", e, file=sys.stderr)

        print("Model loader: finished", flush=True)
    except Exception as e:
        # Print to stdout/stderr so Railway logs capture it
        print("Model loader exception:", e, file=sys.stderr, flush=True)
    finally:
        models_ready = True

@app.on_event("startup")
async def on_startup():
    # Start model loading in background; keep /health fast
    asyncio.create_task(load_models())

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not models_ready or _analyze_frame is None:
        raise HTTPException(status_code=503, detail="Models still loading, try again shortly")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    try:
        result = _analyze_frame(frame)
    except Exception as e:
        print("Analysis error:", e, file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail="Analysis failed")

    return {"results": result}
