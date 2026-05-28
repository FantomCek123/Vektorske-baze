import os
import re
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

INDEX_NAME = "lab-vezba-cloud"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768,
    google_api_key=GEMINI_API_KEY
)

def chunk_screenplay_by_scenes(text):
    """Deli tekst scenarija na logičke celine prema oznakama za scene"""
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

def ingest_to_cloud(putanja_do_fajla):
    """Čita scenario, pretvara scene u Gemini vektore i šalje ih na Pinecone"""
    with open(putanja_do_fajla, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    print("Delim scenario na logičke scene...")
    chunks = chunk_screenplay_by_scenes(raw_text)
    
    vectors_to_upsert = []
    
    for index_num, chunk in enumerate(chunks):
        if len(chunk.strip()) < 15:
            continue
            
        kontekstualni_tekst = f"[Film: Taxi Driver | Scena {index_num + 1}]\n{chunk}"
        print(f"Generišem Gemini cloud vektor za scenu {index_num + 1}/{len(chunks)}...")

        # Generisanje embeddinga preko Gemini API-ja
        vector = embedding_model.embed_query(kontekstualni_tekst)
        
        # Pakovanje u Pinecone format (ID mora biti string)
        point = {
            "id": f"scena_{index_num + 1}",
            "values": vector,
            "metadata": {
                "text": kontekstualni_tekst,
                "film": "Taxi Driver",
                "scena_broj": index_num + 1
            }
        }
        vectors_to_upsert.append(point)
        
    # Slanje podataka na Pinecone Cloud
    print("\nŠaljem vektore na Pinecone...")
    index.upsert(vectors=vectors_to_upsert)
    print(f"Uspelo! Ubačeno {len(vectors_to_upsert)} scena u Pinecone Cloud indeks.")

if __name__ == "__main__":
    putanja_scenarija = "Taxi-Driver-Script.txt" 
    
    if os.path.exists(putanja_scenarija):
        ingest_to_cloud(putanja_scenarija)
    else:
        print(f"Greška: Fajl '{putanja_scenarija}' ne postoji u ovom folderu!")