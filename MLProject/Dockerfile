# Serving image for the retrained Bitcoin price prediction model.
# The model directory is produced by modelling.py (mlflow.sklearn.save_model).
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_MODEL_PATH=/app/model \
    SERVING_PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        mlflow==2.19.0 \
        scikit-learn==1.5.2 \
        pandas==2.2.3 \
        numpy==1.26.4

COPY artifacts/model /app/model

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8080/ping || exit 1

CMD ["mlflow", "models", "serve", "-m", "/app/model", \
     "--host", "0.0.0.0", "--port", "8080", "--env-manager", "local"]
