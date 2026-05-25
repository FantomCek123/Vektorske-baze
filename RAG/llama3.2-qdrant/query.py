import os
import time
import requests
from qdrant_client import QdrantClient
# Vraćamo GoogleEmbeddings samo za potrebe pretrage ove 768-dimenzionalne baze
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Postavljanje API ključa za embedding model
os.environ["GOOGLE_API_KEY"] = "AIzaSyC7AZJbIZO-75VIC4rI17FsHH5VwY8NeRc"

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "reality_is_evil_baza"

print("Inicijalizacija Gemini Embedding modela (768 dimenzija) za lokalni RAG...")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)

def ask_local_rag(user_query):
    # --- MERENJE 1: Pretraga baze (Retrieval) ---
    start_pretraga = time.time()
    
    # Sada generišemo 768 dimenzija, Qdrant će ovo prihvatiti!
    query_vector = embedding_model.embed_query(user_query)
    
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=2,
        with_payload=True
    )
    
    vreme_pretrage = time.time() - start_pretraga
    
    # Izvlačenje konteksta
    context = "\n---\n".join([res.payload["text"] for res in search_results.points])
    
    prompt = f"""Odgovori na pitanje isključivo na osnovu pruženog konteksta. 
Ako u kontekstu nema odgovora, striktno reci da ne znaš. Odgovori na srpskom jeziku.

KONTEKST:
{context}

PITANJE:
{user_query}"""

    # --- MERENJE 2: Generisanje odgovora (Ollama / Llama 3.2) ---
    print(f"\n[LOKALNO] Šaljem prompt lokalnoj Lami...")
    start_generisanje = time.time()
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }, timeout=180)
        
        vreme_generisanja = time.time() - start_generisanje
        odgovor = response.json()["message"]["content"]
        
    except requests.exceptions.ConnectionError:
        return "Greška: Ollama nije pokrenuta!", vreme_pretrage, 0
    except Exception as e:
        return f"Greška: {e}", vreme_pretrage, 0

    return odgovor, vreme_pretrage, vreme_generisanja

if __name__ == "__main__":
    pitanje = "What do scientists say about thermodynamic laws and entropy decay?"
    
    print(f"Pitanje: {pitanje}")
    
    odgovor, t_search, t_gen = ask_local_rag(pitanje)
    
    print("\n" + "="*20 + " LOKALNI AI ODGOVOR (Llama 3.2) " + "="*20)
    print(odgovor)
    print("="*55)
    
    print(f"\n⏱️  BRZINA HIBRIDNO-LOKALNOG SISTEMA:")
    print(f"   • Pretraga (Gemini Embed Cloud + Qdrant): {t_search:.4f} sekundi")
    print(f"   • Generisanje (Ollama Llama 3.2 Lokalno): {t_gen:.4f} sekundi")
    print(f"   • UKUPNO VREME: {t_search + t_gen:.4f} sekundi")