import time
import requests
import streamlit as st
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "taxi_driver_screenplay"

@st.cache_resource(show_spinner=False)
def init_local_clients():
    qdrant_client = QdrantClient(url="http://qdrant-db:6333")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return qdrant_client, embedding_model

def ask_local_rag(user_query):
    qdrant_local, embedding_local = init_local_clients()
    start_pretraga = time.time()
    
    try:
        query_vector = embedding_local.embed_query(user_query)
    except Exception as e:
        return f"Greška pri generisanju lokalnog vektora: {e}", 0, 0

    try:
        search_results = qdrant_local.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=1,  
            with_payload=True
        )
    except Exception as e:
        return f"Greška pri pretrazi Qdrant baze: {e}", 0, 0
    
    vreme_pretrage = time.time() - start_pretraga
    context = "Nema pronađenog konteksta." if not search_results.points else search_results.points[0].payload["text"]
    prompt = f"Odgovori kratko na srpskom jeziku. Kontekst: {context} Pitanje: {user_query} u filmu Taxi Driver"

    start_generisanje = time.time()
    try:
        response = requests.post("http://ollama-service:11434/api/chat", json={
                    "model": "llama3.2:latest", 
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "num_predict": 60,
                        "temperature": 0.1,
                        "num_ctx": 512,       
                        "num_thread": 8,
                    } }, timeout=300)
        vreme_generisanja = time.time() - start_generisanje
        
        res_json = response.json()
        if "message" in res_json and "content" in res_json["message"]:
            odgovor = res_json["message"]["content"]
        elif "response" in res_json:
            odgovor = res_json["response"]
        else:
            odgovor = f"Čudan format odgovora od Ollama-e: {res_json}"
    except Exception as e:
        return f"Greška pri lokalnom generisanju (Ollama): {e}", vreme_pretrage, 0

    return odgovor, vreme_pretrage, vreme_generisanja
