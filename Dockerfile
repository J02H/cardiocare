# CardioCare inference image.
# Build:  docker build -t cardiocare:1.0 .
# Run:    docker run --rm cardiocare:1.0
FROM python:3.12-slim

# Avoid interactive prompts / pyc clutter; unbuffered logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code, model, and data needed for inference.
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Make sure output dir exists at runtime.
RUN mkdir -p outputs logs

# Default: run inference on the bundled sample input.
CMD ["python", "src/inference.py", \
     "--input", "data/sample_input.csv", \
     "--model", "models/final_model.joblib", \
     "--output", "outputs/predictions.csv"]
