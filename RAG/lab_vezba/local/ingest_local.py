import os
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ProductQuantization, ProductQuantizationConfig, CompressionRatio
from langchain_huggingface import HuggingFaceEmbeddings

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "taxi_driver_screenplay"  

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def init_collection():
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Kolekcija '{COLLECTION_NAME}' već postoji. Preskačem kreiranje.")
    else:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            quantization_config=ProductQuantization(
                product=ProductQuantizationConfig(
                    compression=CompressionRatio.X16,
                    always_ram=True
                )
            )
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

def ingest_document(putanja_do_fajla):
    with open(putanja_do_fajla, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    print("Delim scenario na logičke scene...")
    chunks = chunk_screenplay_by_scenes(raw_text)
    
    points = []
    
    for index, chunk in enumerate(chunks):
        if len(chunk.strip()) < 15:
            continue
            
        kontekstualni_tekst = f"[Film: Taxi Driver | Scena {index + 1}]\n{chunk}"
        
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
    print(f"\nUspelo! Ubačeno {len(points)} pametnih scena u Qdrant bazu sa PQ optimizacijom.")

if __name__ == "__main__":
    init_collection()
    
    putanja_scenarija = "Taxi-Driver-Script.txt" 
    
    if os.path.exists(putanja_scenarija):
        print(f"Pronađen fajl {putanja_scenarija}. Pokrećem unos...")
        ingest_document(putanja_scenarija)
    else:
        print(f"Greška: Fajl '{putanja_scenarija}' se ne nalazi u ovom folderu. Proveri putanju!")