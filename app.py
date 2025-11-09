from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
import cv2

# Delay importing prototype9 heavy parts until background load
# from prototype9 import analyze_frame  <- DO NOT import here if it does heavy work

app = FastAPI()

# Allow quick browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint that returns instantly
@app.get("/health")
def health():
    return {"status": "ok"}

# Flags/placeholders for runtime resources
models_ready = False
_analyze_frame = None

# Background loader to import or initialize heavy modules
async def load_models():
    global models_ready, _analyze_frame
    try:
        # Import and initialize heavy resources here so imports don't block server start
        from prototype9 import analyze_frame as af  # imported inside loader
        # If prototype9 does additional heavy work on import, consider moving that work
        # into a function inside prototype9 and call it here instead.
        _analyze_frame = af
        # Example: if you need to load large ML weights, do it here
        # af.load_weights(...)   # pseudo-call if applicable
    except Exception as e:
        # Log error to stdout (Railway logs will show this)
        print("Model load failed:", e)
    models_ready = True

@app.on_event("startup")
async def on_startup():
    # Start loading models in background so /health responds quickly
    asyncio.create_task(load_models())

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models still loading, try again shortly")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    try:
        # Use the lazily-imported analyze function
        result = _analyze_frame(frame)
    except Exception as e:
        # Return a 500 with minimal info and print the error for logs
        print("Analysis error:", e)
        raise HTTPException(status_code=500, detail="Analysis failed")

    return {"results": result}
