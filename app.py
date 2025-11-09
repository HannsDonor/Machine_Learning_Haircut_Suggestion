# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
import cv2
import sys
import time
import traceback
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
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
    status = getattr(response, "status_code", "N/A")
    print(f"REQ END: {request.method} {request.url.path} -> {status} in {duration:.3f}s", flush=True)
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

# /ready shows whether models finished loading
models_ready = False
@app.get("/ready")
def ready():
    return {"ready": models_ready}

# analyze wiring
_analyze_frame = None

# Background loader to import and initialize prototype9 safely
async def load_models(use_stub: bool = False):
    """
    Background initialization.
    - use_stub True installs a minimal stub (fast, low memory) for routing checks.
    - use_stub False runs prototype9.init_models() (may be memory heavy).
    """
    global models_ready, _analyze_frame
    try:
        import prototype9  # must exist in repo

        if use_stub:
            print("Model loader: installing stub (use_stub=True)", flush=True)
            await asyncio.to_thread(prototype9.install_stub_for_testing)
            _analyze_frame = prototype9.analyze_frame
            print("Model loader: stub installed", flush=True)
        else:
            print("Model loader: init_models (use_stub=False)", flush=True)
            await asyncio.to_thread(prototype9.init_models)
            _analyze_frame = prototype9.analyze_frame
            print("Model loader: finished", flush=True)
    except Exception:
        print("Model loader exception:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        models_ready = True
        print(f"Model loader: models_ready={models_ready}", flush=True)

@app.on_event("startup")
async def on_startup():
    # Toggle via env var without editing code
    use_stub_env = os.environ.get("USE_MODEL_STUB", "false").lower()
    use_stub = use_stub_env in ("1", "true", "yes")
    asyncio.create_task(load_models(use_stub=use_stub))

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
        # Run CPU-bound inference in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(_analyze_frame, frame)
    except Exception:
        print("Analysis error:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed")

    return {"results": result}
