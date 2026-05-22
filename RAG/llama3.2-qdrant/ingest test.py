import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings


qdrant = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "moje_beleske"


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def init_collection():
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Kolekcija '{COLLECTION_NAME}' već postoji. Preskačem kreiranje.")
    else:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Uspešno kreirana nova kolekcija: '{COLLECTION_NAME}'")


def chunk_text(text, max_chars=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
        else:
            current_chunk += sentence + ". "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def ingest_document(raw_text):
    chunks = chunk_text(raw_text)
    points = []
    
    for index, chunk in enumerate(chunks):

        vector = embedding_model.embed_query(chunk)
        
        point = PointStruct(
            id=index + 1,
            vector=vector,
            payload={"text": chunk}  # Metadata koji AI zapravo čita
        )
        points.append(point)
        
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ubačeno {len(points)} čankova u Qdrant bazu.")


if __name__ == "__main__":
    init_collection()
    
    interni_dokument = (
        "Projekat 'rtce' koristi MongoDB Atlas kao glavnu bazu. "
        "Za autentifikaciju korisnika koristi se Auth0 sistem. "
        "Rok za završetak prve faze projekta je 15. jun 2026. godine. "
        "Pre puštanja u produkciju, svaki mikroservis mora proći k6 load testiranje."
    )
    
    print("Pokrećem lokalni unos podataka...")
    ingest_document(interni_dokument)