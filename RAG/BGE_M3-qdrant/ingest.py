import os
import re
import time
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model_name = "BAAI/bge-m3"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}

embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "taxi_driver_bge_kolekcija"

def init_collection():
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Brišem staru kolekciju '{COLLECTION_NAME}'...")
        qdrant.delete_collection(collection_name=COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    
def chunk_screenplay_by_scenes(text):
    raw_scenes = re.split(r'(CUT TO:|INT\.|EXT\.)', text)
    chunks = []
    current_chunk = ""
    
    for part in raw_scenes:
        if part in ["CUT TO:", "INT.", "EXT."]:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = part + " "
        else:
            current_chunk += part
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def ingest_document(putanja_do_fajla):
    with open(putanja_do_fajla, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read().strip()

    chunks = chunk_screenplay_by_scenes(raw_text)
    points = []
    print(f"Ukupno pronađeno {len(chunks)} scena. Krećem sa generisanjem vektora...")
    
    start_vreme = time.time()
    
    for index, chunk in enumerate(chunks):
        if len(chunk.strip()) < 15:
            continue
            
        kontekstualni_tekst = f"[Film: Taxi Driver | Scena {index + 1}]\n{chunk}"
        
        if index % 20 == 0: 
            print(f"Indeksiram scenu {index + 1}/{len(chunks)}...")

        vector = embedding_model.embed_query(kontekstualni_tekst)
        
        point = PointStruct(
            id=index + 1,
            vector=vector,
            payload={
                "text": kontekstualni_tekst,
                "film": "Taxi Driver",
                "scena_broj": index + 1
            }
        )
        points.append(point)
        
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
    kraj_vreme = time.time() - start_vreme
    print(f"\nUspelo! Ubačeno {len(points)} pametnih scena u Qdrant bazu.")
    print(f"⏱️ Vreme potrebno za indeksiranje scena: {kraj_vreme:.2f} sekundi.")

if __name__ == "__main__":
    init_collection()
    
    putanja_scenarija = "Taxi-Driver-Script.txt" 
    
    if os.path.exists(putanja_scenarija):
        print(f"Pronađen fajl {putanja_scenarija}. Pokrećem unos...")
        ingest_document(putanja_scenarija)
    else:
        print(f"Greška: Fajl '{putanja_scenarija}' se ne nalazi u ovom folderu. Proveri putanju!")