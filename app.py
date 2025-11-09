from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from prototype9 import analyze_frame

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Read uploaded file into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image data"}

    # Run analysis
    result = analyze_frame(frame)
    return {"results": result}
