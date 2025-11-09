FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure start.py exists and is single entrypoint
COPY start.py .
CMD ["bash", "-lc", "python start.py"]
