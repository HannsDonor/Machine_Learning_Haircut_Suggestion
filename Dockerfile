# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Reduce TF/native threading and disable GPU attempts (help on small instances)
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Minimal OS deps for headless OpenCV and other native libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Optional: pre-initialize models at build time (uncomment if you accept larger image)
# RUN python - <<'PY'
# from prototype9 import init_models
# init_models()
# PY

CMD ["bash", "-lc", "python start.py"]
