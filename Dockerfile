# Multi-stage build for Heart Disease Detection App

# Stage 1: Base image with Python
FROM python:3.10-slim as base

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Backend (FastAPI)
FROM base as backend
COPY ./src /app/src
COPY ./models /app/models
COPY ./data /app/data
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Frontend (Streamlit)
FROM base as frontend
COPY ./app.py /app/
COPY ./templates /app/templates
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]