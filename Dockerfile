# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS dependencies for OpenCV and general usage
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose the port your Flask app will run on
EXPOSE 5000

# Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
