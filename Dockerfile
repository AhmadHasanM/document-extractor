# Multi-stage build untuk optimasi ukuran image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies untuk build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR + language data
    tesseract-ocr \
    tesseract-ocr-ind \
    tesseract-ocr-eng \
    # Poppler untuk PDF
    poppler-utils \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Image processing
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies dari builder
COPY --from=builder /root/.local /root/.local

# Update PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY app/ /app/app/
COPY main.py /app/
COPY .env* /app/

# Create necessary directories
RUN mkdir -p /app/outputs/markdown \
    /app/outputs/images \
    /app/outputs/tables \
    /app/outputs/json \
    /app/uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]