from PIL import Image
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "pretraga_slika"

model = SentenceTransformer('clip-ViT-B-32')

def nadji_slicne_slike(putanja_test_slike):
    img = Image.open(putanja_test_slike)
    
    query_vector = model.encode(img).tolist()
    
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=2,
        with_payload=True
    )
    
    print("\n--- REZULTATI PRETRAGE ---")
    for res in search_results.points:
        skor_slicnosti = res.score
        nadjen_fajla = res.payload["ime_fajla"]
        print(f"Slika: {nadjen_fajla} | Sličnost: {skor_slicnosti:.4f}")

if __name__ == "__main__":
    test_slika = "test_slike/pas_test.jpeg" 
    
    try:
        nadji_slicne_slike(test_slika)
    except FileNotFoundError:
        print(f"Molim te postavi sliku na putanju '{test_slika}' da bi pokrenuo test.")