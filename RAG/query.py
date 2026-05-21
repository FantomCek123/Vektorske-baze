from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests 

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "moje_beleske"

# 1. BESPLATNI LOKALNI EMBEDDING MODEL
# Prvi put će skinuti model od oko 400MB, posle radi oflajn
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ask_rag(user_query):
    query_vector = embedding_model.embed_query(user_query)
    
    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3,
        with_payload=True
    )
    
    context = "\n---\n".join([res.payload["text"] for res in search_results])
    
    prompt = f"Odgovori na pitanje isključivo na osnovu konteksta.\n\nKONTEKST:\n{context}\n\nPITANJE:\n{user_query}"
    
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    
    return response.json()["response"]

if __name__ == "__main__":
    print("--- Pokrećem lokalni RAG sistem za odgovaranje ---")
    
    pitanje = "Koji sistem koristimo za autentifikaciju korisnika?"
    print(f"\nPitanje: {pitanje}")
    
    odgovor = ask_rag_free(pitanje)
    print("\n--- AI ODGOVOR ---")
    print(odgovor)