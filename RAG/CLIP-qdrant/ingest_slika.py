import os
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "pretraga_slika"

print("Učitavam CLIP model...")
model = SentenceTransformer('clip-ViT-B-32')

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

FOLDER_SA_SLIKAMA = "./slike"
points = []

print("Pretvaram slike u vektore...")
for idx, file_name in enumerate(os.listdir(FOLDER_SA_SLIKAMA)):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        putanja_do_slike = os.path.join(FOLDER_SA_SLIKAMA, file_name)
        
        img = Image.open(putanja_do_slike)
        
        vektor_slike = model.encode(img).tolist()
        
        points.append(
            PointStruct(
                id=idx + 1,
                vector=vektor_slike,
                payload={"putanja": putanja_do_slike, "ime_fajla": file_name}
            )
        )

if points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Uspešno indeksirano {len(points)} slika u Qdrant bazu!")
else:
    print("Nema slika u folderu. Ubaci par komada pa pokreni opet.")