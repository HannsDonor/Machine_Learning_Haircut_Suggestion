# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
import cv2
import sys
import time
import traceback

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
    print(f"REQ END: {request.method} {request.url.path} -> {getattr(response,'status_code', 'N/A')} in {duration:.3f}s", flush=True)
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

# Runtime flags/placeholders
models_ready = False
_analyze_frame = None

# Background loader to import and initialize prototype9 safely
async def load_models(use_stub: bool = False):
    """
    Call from startup. If use_stub is True, install a lightweight stub for fast testing.
    """
    global models_ready, _analyze_frame
    try:
        import prototype9

        if use_stub:
            # Install stub for quick testing (no heavy models)
            await asyncio.to_thread(prototype9.install_stub_for_testing)
            _analyze_frame = prototype9.analyze_frame
            print("Model loader: stub installed", flush=True)
        else:
            # Proper init in a thread so event loop is not blocked
            await asyncio.to_thread(prototype9.init_models)
            _analyze_frame = prototype9.analyze_frame
            print("Model loader: finished", flush=True)
    except Exception:
        print("Model loader exception:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        models_ready = True

@app.on_event("startup")
async def on_startup():
    # Toggle use_stub to True while debugging on Railway to confirm routing works
    asyncio.create_task(load_models(use_stub=False))

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
