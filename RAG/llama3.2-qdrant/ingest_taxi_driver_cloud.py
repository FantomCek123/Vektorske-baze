import os
import re
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "taxi_driver_screenplay_cloud_v2"  
API_KEY = "AIzaSyCCDxHT3g2jzUcwGBFta9RzKOZg-Ql5n2A"

def init_collection():
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Kolekcija '{COLLECTION_NAME}' već postoji. Preskačem kreiranje.")
    else:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )
        print(f"Uspešno kreirana nova kolekcija: '{COLLECTION_NAME}'")

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

def get_gemini_embedding(text):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-embedding-001:embedContent?key={API_KEY}"
    payload = {
        "content": {
            "parts": [{"text": text}]
        }
    }
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["embedding"]["values"]

def ingest_document(putanja_do_fajla):
    with open(putanja_do_fajla, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    print("Delim scenario na logičke scene...")
    chunks = chunk_screenplay_by_scenes(raw_text)
    
    points = []
    print(f"Ukupno pronađeno {len(chunks)} scena. Generisanje cloud vektora...")
    
    for index, chunk in enumerate(chunks):
        if len(chunk.strip()) < 15:
            continue
            
        kontekstualni_tekst = f"[Film: Taxi Driver | Scena {index + 1}]\n{chunk}"
         
        print(f"Indeksiram scenu {index + 1}/{len(chunks)}...")

        try:
            vector = get_gemini_embedding(kontekstualni_tekst)
        except Exception as e:
            print(f"Greška na sceni {index + 1}: {e}")
            continue
        
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

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"\nUspelo! Ubačeno {len(points)} scena u cloud kolekciju.")
    else:
        print("\nGreška: Nema generisanih vektora za ubacivanje.")

if __name__ == "__main__":
    init_collection()
    putanja_scenarija = "Taxi-Driver-Script.txt" 
    if os.path.exists(putanja_scenarija):
        ingest_document(putanja_scenarija)
    else:
        print(f"Greška: Fajl '{putanja_scenarija}' nedostaje!")