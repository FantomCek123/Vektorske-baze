from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
import requests

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "taxi_driver_screenplay"


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ask_rag(user_query):
    query_vector = embedding_model.embed_query(user_query)
    
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
        with_payload=True
    )
    
    context = "\n---\n".join([res.payload["text"] for res in search_results.points])
    

    prompt = f"""Odgovori na pitanje isključivo na osnovu pruženog konteksta. 
                Ako u kontekstu nema odgovora, striktno reci da ne znaš i nemoj izmišljati.

                KONTEKST:
                {context}

                PITANJE:
                {user_query}"""

    print("\n")  
    print(prompt)
    print("\n")  

    try:
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "llama3.2",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        })
        
        return response.json()["message"]["content"]
        
    except requests.exceptions.ConnectionError:
        return "Greška: Ollama nije pokrenuta! Pokreni 'ollama run llama3.2' u posebnom terminalu."
    except KeyError:
        return f"Došlo je do greške u formatu odgovora. Ollama je vratila: {response.text}"

if __name__ == "__main__":
    print("--- Pokrećem lokalni RAG sistem za odgovaranje ---")
    
    pitanje = "Kako se zove ženski lik koji radi u kancelariji Charlesa Palantinea i za koji Trevis kaže da je fasciniran njom?"
    print(f"\nPitanje: {pitanje}")
    
    odgovor = ask_rag(pitanje)
    print("\n--- AI ODGOVOR ---")
    print(odgovor)