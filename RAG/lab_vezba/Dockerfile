FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# NOVA CMD KOMANDA SA PROVEROM ZASTAVICE:
CMD sh -c 'if [ ! -f /app/storage/.ingested ]; then \
             python local/ingest_local.py && \
             python cloud/ingest_cloud.py && \
             touch /app/storage/.ingested; \
           else \
             echo "Podaci su već indeksirani. Preskačem ingest skripte."; \
           fi && \
           streamlit run app.py --server.port=8501 --server.address=0.0.0.0'