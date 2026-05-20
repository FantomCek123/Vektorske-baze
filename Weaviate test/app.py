import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)

COLLECTION_NAME = "Filmovi"

try:
    print("1. Proveravam i kreiram kolekciju...")
    
    kolekcija_postoji = client.collections.exists(COLLECTION_NAME)
    
    if not kolekcija_postoji:
        filmovi_kolekcija = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=None,
            properties=[
                wvc.config.Property(name="naslov", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="zanr", data_type=wvc.config.DataType.TEXT),
            ]
        )
        print(" -> Kolekcija je uspešno kreirana!")
    else:
        print(" -> Kolekcija već postoji, preskačem kreiranje.")
        filmovi_kolekcija = client.collections.get(COLLECTION_NAME)

    print("\n2. Ubacujem podatke u Weaviate...")
    filmovi = [
        wvc.data.DataObject(properties={"naslov": "The Matrix", "zanr": "Sci-Fi"}, vector=[0.9, 0.1, 0.0, 0.0]),
        wvc.data.DataObject(properties={"naslov": "Inception", "zanr": "Sci-Fi"}, vector=[0.8, 0.2, 0.0, 0.0]),
        wvc.data.DataObject(properties={"naslov": "The Notebook", "zanr": "Romance"}, vector=[0.0, 0.0, 0.8, 0.9])
    ]

    rezultat_unosa = filmovi_kolekcija.data.insert_many(filmovi)
    print(f" -> Uspešno ubačeno {len(filmovi)} filma.")

    print("\n3. Izvršavam vektorsku pretragu...")
    upit_vektor = [0.85, 0.15, 0.0, 0.0]

    odgovor = filmovi_kolekcija.query.near_vector(
        near_vector=upit_vektor,
        limit=2,
        return_metadata=wvc.query.MetadataQuery(distance=True)
    )

    for objekat in odgovor.objects:
        distanca = objekat.metadata.distance
        print(f" -> Film: {objekat.properties['naslov']} | Distanca: {distanca:.4f}")

    print("\n4. Izvršavam vektorsku pretragu uz filtriranje...")
    filtrirani_odgovor = filmovi_kolekcija.query.near_vector(
        near_vector=upit_vektor,
        limit=2,
        filters=wvc.query.Filter.by_property("zanr").equal("Sci-Fi"),
        return_metadata=wvc.query.MetadataQuery(distance=True)
    )

    for objekat in filtrirani_odgovor.objects:
        print(f" -> [Sci-Fi] Film: {objekat.properties['naslov']} | Distanca: {objekat.metadata.distance:.4f}")

finally:
    client.close()
    print("\nKonekcija zatvorena.")