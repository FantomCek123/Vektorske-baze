import os
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyC7AZJbIZO-75VIC4rI17FsHH5VwY8NeRc"

print("Inicijalizacija Gemini modela...")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "reality_is_evil_baza"

def rag_pretraga_i_odgovor(pitanje, top_k=2):
    pitanje_vektor = embedding_model.embed_query(pitanje)
    
    rezultati_operacije = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=pitanje_vektor,
        limit=top_k
    )
    rezultati = rezultati_operacije.points
    
    if not rezultati:
        print("Nema pronađenih podataka u bazi.")
        return

    pronadjeni_kontekst = ""
    for pogodak in rezultati:
        pronadjeni_kontekst += pogodak.payload.get("text", "") + "\n\n"
        
    print("\nGenerišem odgovor na osnovu pronađenih podataka...")
    
    prompt = f"""
    Ti si koristan asistent. Odgovori na korisnikovo pitanje isključivo koristeći pruženi kontekst iz teksta. 
    Ako u kontekstu nema odgovora, reci da ne znaš. Budu precizan i odgovori na srpskom jeziku.

    KONTEKST IZ BAZE:
    {pronadjeni_kontekst}

    KORISNIKOVO PITANJE:
    {pitanje}

    ODGOVOR:
    """
    
    odgovor_modela = llm.invoke(prompt)
    
    print("\n" + "="*50)
    print("DIREKTAN ODGOVOR:")
    print("="*50)
    print(odgovor_modela.content)
    print("="*50)

if __name__ == "__main__":
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Greška: Kolekcija '{COLLECTION_NAME}' ne postoji!")
    else:
        test_pitanje = "What do scientists say about thermodynamic laws and entropy decay?"
        rag_pretraga_i_odgovor(test_pitanje, top_k=2)