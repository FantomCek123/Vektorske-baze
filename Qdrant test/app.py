from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(
    url="http://localhost:6333", 
    prefer_grpc=False
)

COLLECTION_NAME = "filmovi_rest"

def main():
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print("1. Kolekcija ne postoji. Kreiram kolekciju preko REST API-ja (HTTP)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
    else:
        print("1. Kolekcija već postoji u bazi, preskačem kreiranje.")

    print("2. Ubacujem podatke u JSON formatu...")
    filmovi = [
        PointStruct(id=1, vector=[0.9, 0.1, 0.0, 0.0], payload={"naslov": "The Matrix", "zanr": "Sci-Fi"}),
        PointStruct(id=2, vector=[0.8, 0.05, 0.1, 0.1], payload={"naslov": "Inception", "zanr": "Sci-Fi"}),
        PointStruct(id=3, vector=[0.0, 0.0, 0.8, 0.9], payload={"naslov": "The Notebook", "zanr": "Romance"})
    ]


    client.upsert(collection_name=COLLECTION_NAME, wait=True, points=filmovi)

    print("\n3. Izvršavam REST HTTP vektorsku pretragu...")
    upit_vektor = [0.3, 0.0, 0.8, 0.9]


    odgovor = client.query_points(
        collection_name=COLLECTION_NAME,
        query=upit_vektor,
        limit=2  
    )

    for pogodak in odgovor.points:
        print(f" -> Film: {pogodak.payload['naslov']} | Sličnost: {pogodak.score:.4f}")

    print("\n4. Izvršavam pretragu uz napredno filtriranje metapodataka...")
    filtrirani_odgovor = client.query_points(
        collection_name=COLLECTION_NAME,
        query=upit_vektor,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="zanr",
                    match=MatchValue(value="Sci-Fi")
                )
            ]
        ),
        limit=2
    )

    for pogodak in filtrirani_odgovor.points:
        print(f" -> [Sci-Fi] Film: {pogodak.payload['naslov']} | Sličnost: {pogodak.score:.4f}")

if __name__ == "__main__":
    main()