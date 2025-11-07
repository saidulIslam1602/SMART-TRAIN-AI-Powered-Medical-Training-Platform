# SMART-TRAIN AI Platform - Production Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Labels for image metadata (industry standard)
LABEL org.opencontainers.image.title="SMART-TRAIN AI Platform"
LABEL org.opencontainers.image.description="AI-Powered Medical Training Platform"
LABEL org.opencontainers.image.version=${VERSION}
LABEL org.opencontainers.image.created=${BUILD_DATE}
LABEL org.opencontainers.image.revision=${VCS_REF}
LABEL org.opencontainers.image.vendor="SMART-TRAIN"
LABEL org.opencontainers.image.licenses="MIT"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements_core.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_core.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r smarttrain && useradd -r -g smarttrain smarttrain

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml setup.cfg ./

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed compliance/audit_logs && \
    chown -R smarttrain:smarttrain /app

# Switch to non-root user
USER smarttrain

# Set environment variables
ENV PYTHONPATH=/app/src
ENV SMART_TRAIN_LOG_DIR=/app/logs
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from smart_train.core.config import get_config; get_config()" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "smart_train.api.main"]
