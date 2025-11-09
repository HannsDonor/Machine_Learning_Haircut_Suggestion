# Use lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, TensorFlow, and Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 ffmpeg libsm6 libxext6 git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Flask app files
COPY . .

# Expose the default port
EXPOSE 5000

# Command for Railway / gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
