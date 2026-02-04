# Backend Dockerfile for IGQK SaaS Platform
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy IGQK library
COPY ../igqk /app/igqk

# Install IGQK
RUN pip install -e /app/igqk

# Copy backend code
COPY . /app/backend

# Create necessary directories
RUN mkdir -p /app/backend/compressed_models \
    /app/backend/models_cache \
    /app/backend/datasets_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run backend
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
