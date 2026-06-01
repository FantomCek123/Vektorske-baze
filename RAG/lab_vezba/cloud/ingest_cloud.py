import os
import re
import time
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
    with open(putanja_do_fajla, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    print("Delim scenario na logičke scene...")
    chunks = chunk_screenplay_by_scenes(raw_text)
    
    vectors_to_upsert = []
    ukupan_broj_scena = len(chunks)
    
    print("Pokrećem slanje scena JEDNU PO JEDNU sa pametnim re-try mehanizmom...\n")
    
    for index_num, chunk in enumerate(chunks):
        if len(chunk.strip()) < 15:
            continue
            
        scena_id_broj = index_num + 1
        kontekstualni_tekst = f"[Film: Taxi Driver | Scena {scena_id_broj}]\n{chunk}"
        
        time.sleep(0.5)
        
        vreme_cekanja = 2  # Počinjemo od 2 sekunde
        maksimalno_cekanje = 32
        uspesno_generisano = False
        vector = None
        
        while not uspesno_generisano:
            try:
                print(f"Generišem Gemini cloud vektor za scenu {scena_id_broj}/{ukupan_broj_scena}...")
                vector = embedding_model.embed_query(kontekstualni_tekst)
                uspesno_generisano = True
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if vreme_cekanja <= maksimalno_cekanje:
                        print(f"Geska kod scene {scena_id_broj}. Čekam {vreme_cekanja}s pre ponovnog pokušaja...")
                        time.sleep(vreme_cekanja)
                        vreme_cekanja *= 2  # Dupliramo vreme (2 -> 4 -> 8 -> 16 -> 32)
                    else:
                        print(f"❌ Limit od 32 sekunde prekoračen na sceni {scena_id_broj}. Prekidam.")
                        raise e
                else:
                    print(f"❌ Neočekivana greška: {e}")
                    raise e
                    
        point = {
            "id": f"scena_{scena_id_broj}",
            "values": vector,
            "metadata": {
                "text": kontekstualni_tekst,
                "film": "Taxi Driver",
                "scena_broj": scena_id_broj
            }
        }
        vectors_to_upsert.append(point)
        
    print(f"\nSve scene su uspešno pretvorene u vektore! Šaljem ukupno {len(vectors_to_upsert)} vektora na Pinecone...")
    index.upsert(vectors=vectors_to_upsert)
    print(f"Uspelo! Ubačeno {len(vectors_to_upsert)} scena u Pinecone Cloud indeks.")

if __name__ == "__main__":
    putanja_scenarija = "Taxi-Driver-Script.txt" 
    if os.path.exists(putanja_scenarija):
        ingest_to_cloud(putanja_scenarija)
    else:
        print(f"Greška: Fajl '{putanja_scenarija}' ne postoji!")