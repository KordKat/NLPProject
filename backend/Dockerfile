# STEM OS backend (GPU-capable). Build context: ./backend
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libffi8 libjpeg-turbo8 \
        fonts-dejavu-core pandoc curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY app ./app

RUN useradd -m -u 1000 stemos && mkdir -p /storage /models && chown -R stemos /storage /models /app
USER stemos

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
