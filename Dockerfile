# Minimal image for the live regime endpoint (Railway / Render / Fly).
FROM python:3.11-slim

WORKDIR /app

# install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY regimelab ./regimelab
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps .

# PaaS platforms inject $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# shell form so $PORT expands at runtime
CMD uvicorn regimelab.api:app --host 0.0.0.0 --port ${PORT}
