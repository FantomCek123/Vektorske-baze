# qdrant_crud.py
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "taxi_driver_screenplay"

@st.cache_resource
def get_qdrant_resources():
    client = QdrantClient(url="http://qdrant_local:6333")
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return client, model

def create_or_update_qdrant(scena_id: int, tekst: str, film: str = "Taxi Driver") -> bool:
    """[CREATE / UPDATE] - Vraća True ako je upsert uspešan."""
    try:
        qdrant_client, hf_model = get_qdrant_resources()
        vektor = hf_model.embed_query(tekst)
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=scena_id,
                    vector=vektor,
                    payload={"text": tekst, "film": film}
                )
            ]
        )
        return True
    except Exception as e:
        print(f"Qdrant Upsert Greška: {e}")
        return False

def read_qdrant(scena_id: int):
    qdrant_client, _ = get_qdrant_resources()
    try:
        res = qdrant_client.retrieve(collection_name=COLLECTION_NAME, ids=[scena_id])
        if res:
            return res[0].payload
    except Exception as e:
        print(f"Qdrant Read Greška: {e}")
    return None

def delete_qdrant(scena_id: int) -> bool:
    try:
        qdrant_client, _ = get_qdrant_resources()
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[scena_id]
        )
        return True
    except Exception as e:
        print(f"Qdrant Delete Greška: {e}")
        return False