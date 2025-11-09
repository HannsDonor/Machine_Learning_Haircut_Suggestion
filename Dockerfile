# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
# --use-feature=fast-deps can speed up installation for many packages
RUN pip install --no-cache-dir --use-feature=fast-deps -r requirements.txt

# Copy app code
COPY . .

# Expose port (if your app uses Flask/Streamlit)
EXPOSE 8501

# Default command (example: Streamlit)
CMD ["streamlit", "run", "app.py"]
