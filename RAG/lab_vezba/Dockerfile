FROM python:3.10-slim

WORKDIR /app

# Instaliramo samo osnovne alate za kompajliranje koji su neophodni za Python biblioteke
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD python local/ingest_local.py && python cloud/ingest_cloud.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0