import os
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ProductQuantization, ProductQuantizationConfig, CompressionRatio
from langchain_huggingface import HuggingFaceEmbeddings  # PROMENJENO

qdrant = QdrantClient(url="http://qdrant-db:6333")
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
        print(f"Uspešno kreirana nova lokalna kolekcija: '{COLLECTION_NAME}'")

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
    
    svi_tekstovi = []
    validni_indeksi = [] 
    
    for index, chunk in enumerate(chunks):
        if len(chunk.strip()) < 15:
            continue
            
        kontekstualni_tekst = f"\n{chunk}"
        svi_tekstovi.append(kontekstualni_tekst)
        validni_indeksi.append(index + 1)

    print(f"Pripremljeno {len(svi_tekstovi)} scena za lokalno indeksiranje.")
    print("Generišem vektore lokalno preko HuggingFace modela...")
    
    svi_vektori = embedding_model.embed_documents(svi_tekstovi)
    
    print("Vektori uspešno generisani. Pakujem podatke za Qdrant...")
    points = []
    
    for i, vector in enumerate(svi_vektori):
        point = PointStruct(
            id=validni_indeksi[i],
            vector=vector,
            payload={
                "text": svi_tekstovi[i],
                "film": "Taxi Driver",
            }
        )
        points.append(point)
        
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\nUspelo! Ubačeno {len(points)} scena u lokalnu Qdrant bazu.")

if __name__ == "__main__":
    init_collection()
    
    putanja_scenarija = "Taxi-Driver-Script.txt" 
    
    if os.path.exists(putanja_scenarija):
        ingest_document(putanja_scenarija)
    else:
        print(f"Greška: Fajl '{putanja_scenarija}' se ne nalazi u ovom folderu.")