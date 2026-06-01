# pinecone_crud.py
import os
import streamlit as st
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

INDEX_NAME = "lab-vezba-cloud"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

@st.cache_resource
def get_pinecone_resources():
    """
    Inicijalizuje Pinecone indeks i Gemini model koristeći 
    globalne API ključeve i Streamlit keširanje.
    """
    # Popravljena sintaksna greška - koristimo direktno globalne ključeve sa vrha fajla
    model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768,
        google_api_key=GEMINI_API_KEY
    )
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME) 
    
    return index, model

def create_or_update_pinecone(scena_id: int, tekst: str, film: str = "Taxi Driver") -> bool:
    """[CREATE / UPDATE] - Vraća True ako je upis (upsert) u Pinecone uspeo."""
    try:
        pinecone_index, gemini_model = get_pinecone_resources()
        vektor = gemini_model.embed_query(tekst)
        
        odgovor = pinecone_index.upsert(
            vectors=[
                {
                    "id": f"scena_{scena_id}",
                    "values": vektor,
                    "metadata": {"text": tekst, "film": film, "scena_broj": scena_id}
                }
            ]
        )
        # Pinecone vraća rečnik sa brojem upisanih stavki, npr. {"upserted_count": 1}
        return odgovor.get("upserted_count", 0) > 0
    except Exception as e:
        print(f"Pinecone Upsert Greška: {e}")
        return False

def read_pinecone(scena_id: int):
    """[READ] - Traži vektor u Pinecone-u preko string ID-ja i vraća metapodatke ili None."""
    pinecone_index, _ = get_pinecone_resources()
    string_id = f"scena_{scena_id}"
    try:
        res = pinecone_index.fetch(ids=[string_id])
        if string_id in res["vectors"]:
            return res["vectors"][string_id]["metadata"]
    except Exception as e:
        print(f"Pinecone Read Greška: {e}")
    return None

def delete_pinecone(scena_id: int) -> bool:
    """[DELETE] - Briše vektor iz Pinecone-a. Vraća True ako prođe bez greške."""
    try:
        pinecone_index, _ = get_pinecone_resources()
        pinecone_index.delete(ids=[f"scena_{scena_id}"])
        return True
    except Exception as e:
        print(f"Pinecone Delete Greška: {e}")
        return False