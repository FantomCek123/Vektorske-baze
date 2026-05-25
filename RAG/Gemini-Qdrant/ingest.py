import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = "AIzaSyC7AZJbIZO-75VIC4rI17FsHH5VwY8NeRc"

print("Inicijalizacija Gemini Embedding modela...")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "reality_is_evil_baza"

FAJL = "Reality is evil.txt"

if not os.path.exists(FAJL):
    print(f"Greška: Fajl '{FAJL}' mora biti u istom folderu!")
    exit()

print(f"Učitavam fajl '{FAJL}' i delim ga na pasuse...")
with open(FAJL, "r", encoding="utf-8", errors="ignore") as f:
    linije = f.readlines()

pasusi = []
trenutni_pasus = ""
for linija in linije:
    if linija.strip():
        trenutni_pasus += linija.strip() + " "
    else:
        if trenutni_pasus.strip():
            pasusi.append(trenutni_pasus.strip())
            trenutni_pasus = ""
if trenutni_pasus.strip():
    pasusi.append(trenutni_pasus.strip())

ukupno_pasusa = len(pasusi)
print(f"Ukupno kreirano {ukupno_pasusa} pasusa za indeksiranje.")

if qdrant.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Brišem staru kolekciju '{COLLECTION_NAME}'...")
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
print("Nova čista kolekcija je spremna.")

print("\nPokrećem unos: šaljem pasus po pasus...")

points = []
for index, pasus in enumerate(pasusi):
    print(f"Procesiram pasus {index + 1} / {ukupno_pasusa}...")
    
    try:
        vektor = embedding_model.embed_query(pasus)
        
        point = PointStruct(
            id=index + 1,
            vector=vektor,
            payload={
                "text": pasus,
                "paragraf_id": index + 1,
                "izvor": "Reality is evil"
            }
        )
        points.append(point)
        
            
    except Exception as e:
        print(f"Greška na pasusu {index + 1}: {e}")
        print("Pauziram 10 sekundi da se Google server oporavi...")
        time.sleep(10)

print("\nUpisujem sve uspešno generisane vektore u lokalni Qdrant...")
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"[USPEH] Završeno! Ubačeno {len(points)} tačaka u Qdrant.")