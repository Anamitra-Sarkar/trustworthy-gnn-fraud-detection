FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/model

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html && \
    pip install torch-geometric==2.5.0 && \
    pip install -r requirements.txt

RUN mkdir -p /model

COPY . .

RUN useradd -m -u 1000 user && \
    chown -R user:user /app /model
USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
