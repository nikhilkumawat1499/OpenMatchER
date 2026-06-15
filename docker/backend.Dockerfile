FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/backend:/app/matching:/app/pipelines:/app/llm:/app/embeddings

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt psycopg[binary]==3.2.3

COPY backend /app/backend
COPY matching /app/matching
COPY pipelines /app/pipelines
COPY llm /app/llm
COPY embeddings /app/embeddings

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]

