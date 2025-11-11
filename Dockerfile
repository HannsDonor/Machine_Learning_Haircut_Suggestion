# -----------------------
# Base image: slim, Python 3.11
# -----------------------
FROM python:3.11-slim

# -----------------------
# Environment variables
# -----------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PORT=8080

# -----------------------
# Minimal OS dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Working directory
# -----------------------
WORKDIR /app

# -----------------------
# Copy requirements and install Python dependencies
# -----------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install deepface==0.0.93 && \
    pip show deepface

# -----------------------
# Pre-download DeepFace retinaface.h5 weights
# -----------------------
RUN mkdir -p /root/.deepface/weights && \
    curl -L -o /root/.deepface/weights/retinaface.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5

# -----------------------
# Copy application code
# -----------------------
COPY . .

# -----------------------
# Expose port
# -----------------------
EXPOSE 8080

# -----------------------
# Start FastAPI app
# -----------------------
CMD ["python", "start.py"]
