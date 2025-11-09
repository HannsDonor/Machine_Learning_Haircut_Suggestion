# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port (Railway will map it)
EXPOSE 5000

# Command to run
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
