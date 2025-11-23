# Dockerfile 
FROM python:3.10-slim

# ENV DEBIAN_FRONTEND=noninteractive
ENV RUN_ARTIFACTS=0
WORKDIR /app

# system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy project files
COPY . /app

# make start.sh executable
RUN chmod +x /app/start.sh

# expose port for FastAPI/uvicorn
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# default entry: run start script (which runs preprocess if needed then uvicorn)
CMD ["./start.sh"]

