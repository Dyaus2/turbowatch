# TurboWatch — single image serving the FastAPI backend + built React dashboard (Hugging Face Space, Docker SDK)
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TURBOWATCH_ROOT=/app OMP_NUM_THREADS=2 MLFLOW_DISABLE_AGENT_HINT=1
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt ./
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .
COPY artifacts/ artifacts/
COPY results/ results/
COPY --from=ui /ui/dist frontend/dist
# public NASA C-MAPSS data, fetched at build time (not redistributed in the repo)
RUN python -c "from turbowatch.data import download; download('/app/data/raw')" && chown -R user:user /app
USER user
EXPOSE 7860
CMD ["uvicorn", "turbowatch.api:app", "--host", "0.0.0.0", "--port", "7860"]
