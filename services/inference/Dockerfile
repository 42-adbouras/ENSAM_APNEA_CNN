# ── Stage 1: build the web interface ─────────────────────────────────────────

FROM node:22-slim AS web

WORKDIR /web

# Install the npm packages, at the exact versions of package-lock.json
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

# Check the TypeScript and build the interface into /web/dist
COPY web/ ./
RUN npm run build


# ── Stage 2: the inference service ───────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install the Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add the service's code, and the interface built in stage 1
COPY source/ ./
COPY --from=web /web/dist static/

# Run as a non-root user
RUN useradd --system --no-create-home inference
USER inference

# Listen on port 8000; Docker checks /health every 30 seconds
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
