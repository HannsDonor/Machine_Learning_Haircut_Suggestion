# -----------------------
# Base image: slim, Python 3.10
# -----------------------
FROM python:3.10-slim

# -----------------------
# Environment variables
# -----------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# TensorFlow CPU only, low memory/thread usage
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Hugging Face expects the app to listen on port 8080
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
# Install Python dependencies
# -----------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
# Expose port (HF Spaces will bind 8080)
# -----------------------
EXPOSE 8080

# -----------------------
# Start FastAPI app
# -----------------------
CMD ["python", "start.py"]

