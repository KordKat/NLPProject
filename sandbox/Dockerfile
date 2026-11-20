FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
RUN pip install fastapi "uvicorn[standard]" pydantic numpy scipy sympy pandas matplotlib python-pptx openpyxl \
    && useradd -m -u 1000 runner

WORKDIR /srv
COPY server.py .
USER 1000:1000
EXPOSE 8090
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
