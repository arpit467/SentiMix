# Use lightweight Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheel first for ultra-fast download and small image footprint
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure results directory exists
RUN mkdir -p results/saved_models

# Expose standard port
EXPOSE 7860

# Default command starts the FastAPI server (serving static dashboard + API)
# To run Streamlit instead, override CMD: ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
