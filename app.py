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
        # Import the module (lightweight) and run its blocking init off the event loop
        import prototype9

        # For quick debugging you can use the stub:
        # await asyncio.to_thread(prototype9.install_stub_for_testing)
        # _analyze_frame = prototype9.analyze_frame

        # Proper init (runs heavy work in a thread so event loop isn't blocked)
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
        # Run inference in a thread if analyze_frame is CPU-bound to avoid blocking
        result = await asyncio.to_thread(_analyze_frame, frame)
    except Exception:
        print("Analysis error:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed")

    return {"results": result}
