import os
import time
import requests
import streamlit as st
from pinecone import Pinecone
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  # Vraćeno: Uvoz za .env

# Učitavanje promenljivih iz .env fajla
load_dotenv()

INDEX_NAME = "lab-vezba-cloud"
COLLECTION_NAME = "taxi_driver_screenplay"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="Filmski RAG Asistent", layout="wide")

@st.cache_resource
def init_local_clients():
    qdrant_client = QdrantClient(url="http://localhost:6333")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return qdrant_client, embedding_model

@st.cache_resource
def init_cloud_clients():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_client = pc.Index(INDEX_NAME)
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768,
        google_api_key=GEMINI_API_KEY
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )
    return index_client, embedding_model, llm

with st.sidebar:
    st.title("Podešavanja")
    rezim_rada = st.radio(
        "Izaberi model pretrage:",
        ("Local", "Cloud"),
    )
    st.markdown("---")
    placeholder_stats = st.empty()

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
    prompt = f"Odgovori kratko na srpskom jeziku. Kontekst: {context} Pitanje: {user_query}"

    start_generisanje = time.time()
    try:
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 50,  
                "temperature": 0.1,
                "num_ctx": 1024
            }
        }, timeout=300)
        vreme_generisanja = time.time() - start_generisanje
        odgovor = response.json()["message"]["content"]
    except Exception as e:
        return f"Greška pri lokalnom generisanju (Ollama): {e}", vreme_pretrage, 0

    return odgovor, vreme_pretrage, vreme_generisanja

def ask_cloud_rag(user_query):
    index_client, embedding_cloud, llm = init_cloud_clients()
    start_pretraga = time.time()
    
    try:
        query_vector = embedding_cloud.embed_query(user_query)
    except Exception as e:
        return f"Greška pri generisanju cloud vektora: {e}", 0, 0

    try:
        search_results = index_client.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True
        )
    except Exception as e:
        return f"Greška pri pretrazi Pinecone baze: {e}", 0, 0
    
    vreme_pretrage = time.time() - start_pretraga
    context = "Nema pronađenog konteksta." if not search_results.matches else search_results.matches[0].metadata["text"]
    prompt = f"Odgovori kratko na srpskom jeziku. Kontekst: {context} Pitanje: {user_query}"

    start_generisanje = time.time()
    try:
        response = llm.invoke(prompt)
        vreme_generisanja = time.time() - start_generisanje
        odgovor = response.content
    except Exception as e:
        return f"Greška pri generisanju preko Gemini API: {e}", vreme_pretrage, 0

    return odgovor, vreme_pretrage, vreme_generisanja

left_pad, main_col, right_pad = st.columns([1, 2, 1])

with main_col:
    st.title("Taxi Driver chatbot")
    st.markdown("---")
    input_placeholder = st.container()

with input_placeholder:
    user_pitanje = st.chat_input("Postavi pitanje...")

if user_pitanje:
    with main_col:
        with st.chat_message("user"):
            st.write(user_pitanje)
            
        with st.chat_message("assistant"):
            with st.spinner("AI analizira..."):
                if rezim_rada == "Local":
                    odgovor, t_search, t_gen = ask_local_rag(user_pitanje)
                else:
                    odgovor, t_search, t_gen = ask_cloud_rag(user_pitanje)
                st.write(odgovor)
        
    with placeholder_stats.container():
        st.markdown("### Statistika poslednjeg upita")
        st.metric("Pretraga (Vektori)", f"{t_search:.3f} s")
        st.metric("Generisanje (Model)", f"{t_gen:.3f} s")
        st.metric("Ukupno", f"{t_search + t_gen:.3f} s")

st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        left: 0 !important;
        right: 0 !important;
        margin: auto !important;
        width: 50% !important; 
    }
    </style>
""", unsafe_allow_html=True)