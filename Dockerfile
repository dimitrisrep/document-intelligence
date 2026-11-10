FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY artifacts ./artifacts
USER 65534:65534
EXPOSE 8003
CMD ["uvicorn", "document_intelligence.api:app", "--host", "0.0.0.0", "--port", "8003"]
