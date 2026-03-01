# ============================================================
# WorldLens — Multi-stage Docker Build
# ============================================================
# Stage 1: Build the React frontend
# Stage 2: Run the Python backend (serves frontend static files too)
# ============================================================

# -------------------------------------------------------
# Stage 1: Frontend build
# -------------------------------------------------------
FROM --platform=linux/amd64 node:22-slim AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (cache layer)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

# Copy source and build
COPY frontend/ ./

# Build with default backend URL (overridable at runtime via env)
ARG VITE_BACKEND_URL=http://localhost:8000
ARG VITE_STREAM_API_KEY=""
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}
ENV VITE_STREAM_API_KEY=${VITE_STREAM_API_KEY}

RUN npm run build

# -------------------------------------------------------
# Stage 2: Backend + serve frontend static files
# -------------------------------------------------------
FROM --platform=linux/amd64 python:3.12-slim AS runtime

# System deps for OpenCV, MediaPipe, and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy backend dependency files first (cache layer)
COPY backend/pyproject.toml backend/uv.lock backend/requirements.txt backend/.python-version ./backend/

WORKDIR /app/backend

# Install Python dependencies
RUN uv sync --no-dev

# Copy the rest of the backend (includes yolo11n.pt and yolo11n-pose.pt)
COPY backend/ ./

# Copy built frontend into backend's static serving directory
COPY --from=frontend-build /app/frontend/dist /app/frontend-dist

# Copy env example for reference
COPY backend/.env.example /app/.env.example

# Expose backend port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/mode || exit 1

# Default command — serve backend on 0.0.0.0:8000
# Users must provide their own .env file or environment variables
CMD ["uv", "run", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
