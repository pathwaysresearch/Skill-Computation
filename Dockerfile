FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    APP_MODE=streamlit

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app
RUN mkdir -p /app/runtime/uploads /app/runtime/artifacts && chmod +x /app/docker/start.sh

EXPOSE 8080

CMD ["/app/docker/start.sh"]
