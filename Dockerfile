# Use a Python 3.11 base
FROM python:3.11-slim

WORKDIR /app

# Copy your requirements.txt
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install main ML packages first
RUN pip install --no-cache-dir tensorflow==2.14.0 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# Then the others
RUN pip install --no-cache-dir \
    mediapipe>=0.10.8 \
    deepface>=0.0.95 \
    mtcnn>=1.0.0 \
    retina-face>=0.0.17 \
    opencv-python>=4.12.0 \
    opencv-contrib-python>=4.12.0 \
    pandas>=2.3.3 \
    scipy>=1.8.0 \
    scikit-learn>=1.7.2 \
    joblib>=1.4.2 \
    tqdm>=4.67.1 \
    flask>=3.1.2 \
    flask-cors>=6.0.1 \
    gunicorn>=23.0.0 \
    matplotlib>=3.10.7 \
    Pillow>=12.0.0 \
    gdown>=5.2.0 \
    sympy>=1.14.0 \
    networkx>=3.5 \
    fsspec>=2025.10.0 \
    sounddevice>=0.4.4 \
    lz4>=4.3.3 \
    filelock>=3.20.0 \
    ml-dtypes>=0.2.0
