# Use a lightweight Python base image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the container listens on the PORT Railway provides by running start.py
# start.py should call uvicorn with access logs and read PORT from env
COPY start.py .
CMD ["bash", "-lc", "python start.py"]
