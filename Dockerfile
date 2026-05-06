# =============================================================================
# Predictive Maintenance API — Dockerfile
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder) — install Python dependencies into a virtual environment
#   Stage 2 (runtime) — copy only the venv + source; no build tools in final image
#
# Build:   docker build -t predictive-maintenance-api .
# Run:     docker run -p 8000:8000 predictive-maintenance-api
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed by some ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment inside the image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install production dependencies (skip dev-only ones)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.111.0 \
        uvicorn[standard]==0.29.0 \
        pydantic==2.7.1 \
        pandas>=1.5.0 \
        numpy>=1.23.0 \
        scikit-learn>=1.2.0 \
        joblib>=1.2.0 \
        imbalanced-learn>=0.10.0 \
        xgboost>=1.7.0 \
        lightgbm>=3.3.5 \
        catboost>=1.1.0 \
        gdown>=4.6.3

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="your-email@example.com"
LABEL description="Predictive Maintenance API — LightGBM champion model"
LABEL version="1.0.0"

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Runtime system libs only (LightGBM needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/     ./src/
COPY api/     ./api/
COPY artifacts/ ./artifacts/

# Ensure the appuser owns the working directory
RUN chown -R appuser:appgroup /app

USER appuser

# Expose the API port
EXPOSE 8000

# Health check — Docker will mark the container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]