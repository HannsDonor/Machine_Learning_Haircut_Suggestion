from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
import cv2
import sys
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware to trace incoming requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    print(f"REQ START: {request.method} {request.url.path}", flush=True)
    try:
        response = await call_next(request)
    except Exception as e:
        print(f"REQ ERROR: {request.method} {request.url.path} -> EXC {e}", flush=True)
        raise
    duration = time.time() - start
    print(f"REQ END: {request.method} {request.url.path} -> {response.status_code} in {duration:.3f}s", flush=True)
    return response

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
        from prototype9 import analyze_frame as af
        _analyze_frame = af

        # If prototype9 requires explicit init, call it here (example):
        # if hasattr(af, "init"):
        #     af.init()

        print("Model loader: finished", flush=True)
    except Exception as e:
        print("Model loader exception:", e, file=sys.stderr, flush=True)
    finally:
        models_ready = True

@app.on_event("startup")
async def on_startup():
    # Start model loading in background so /health responds immediately
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
