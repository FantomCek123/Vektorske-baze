import urllib.parse
import time
from pymongo import MongoClient

sifra = "vorkraft1"
bezbedna_sifra = urllib.parse.quote_plus(sifra)
MONGO_URI = f"mongodb+srv://stricojebci:{bezbedna_sifra}@cluster0.hamavtb.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)
db = client["VektorskeBaze"]  
collection = db["movies"]   

def main():
    print("1. Čistim stare podatke i ubacujem nove...")
    collection.delete_many({}) 
    
    filmovi = [
        {"naslov": "The Matrix", "zanr": "Sci-Fi", "embedding": [0.9, 0.1, 0.0, 0.0]},
        {"naslov": "Inception", "zanr": "Sci-Fi", "embedding": [0.8, 0.05, 0.1, 0.1]},
        {"naslov": "The Notebook", "zanr": "Romance", "embedding": [0.0, 0.0, 0.8, 0.9]}
    ]
    
    collection.insert_many(filmovi)
    print(f" -> Uspešno ubačeno {len(filmovi)} filma.")
    
    print(" -> Čekam 20 sekundi da Atlas u klaudu osveži HNSW indeks...")
    time.sleep(20)

    print("\n2. Izvršavam vektorsku pretragu preko $vectorSearch agregacije...")
    upit_vektor = [0.9, 0.2, 0.0, 0.0]

    pipeline_pretraga = [
        {
            "$vectorSearch": {
                "index": "default", 
                "path": "embedding",
                "queryVector": upit_vektor,
                "numCandidates": 10, 
                "limit": 2
            }
        },
        {
            "$project": {
                "_id": 0,
                "naslov": 1,
                "zanr": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    rezultati = list(collection.aggregate(pipeline_pretraga))
    if not rezultati:
        print(" -> Atlas je vratio praznu listu dokumenata.")
    for film in rezultati:
        print(f" -> Film: {film['naslov']} | Sličnost: {film['score']:.4f}")

    print("\n3. Izvršavam vektorsku pretragu uz PRE-FILTERING...")
    
    pipeline_sa_filterom = [
        {
            "$vectorSearch": {
                "index": "default",
                "path": "embedding",
                "queryVector": upit_vektor,
                "numCandidates": 10,
                "limit": 2,
                "filter": {"zanr": {"$eq": "Sci-Fi"}} 
            }
        },
        {
            "$project": {
                "_id": 0,
                "naslov": 1,
                "zanr": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    rezultati_filter = list(collection.aggregate(pipeline_sa_filterom))
    if not rezultati_filter:
        print(" -> Atlas je vratio praznu listu za filtriranu pretragu.")
    for film in rezultati_filter:
        print(f" -> [Sci-Fi] Film: {film['naslov']} | Sličnost: {film['score']:.4f}")

if __name__ == "__main__":
    main()