# Use Python 3.11 slim — torch 2.2.2 is confirmed available for this version
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements_deploy.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and weights
COPY app/app.py ./app.py
COPY src/model_unet.py ./src/model_unet.py
COPY checkpoints/pix2pix_best.pth ./checkpoints/pix2pix_best.pth

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Launch the app
CMD ["python", "app.py"]
