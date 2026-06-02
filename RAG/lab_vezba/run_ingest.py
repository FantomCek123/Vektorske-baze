import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


FILE_PATH     = "Taxi-Driver-Script.txt"
CATEGORY_NAME = "Taxi Driver"

from services.document_service import chunk_document_by_delimiters

def _init_qdrant_collection(db) -> None:
    if not db.client.collection_exists(collection_name=db.collection_name):
        from qdrant_client.models import VectorParams, Distance, ProductQuantization, ProductQuantizationConfig, CompressionRatio
        print(f"🔧 [Qdrant] Kreiram lokalnu kolekciju: '{db.collection_name}'...")
        db.client.create_collection(
            collection_name=db.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            quantization_config=ProductQuantization(
                product=ProductQuantizationConfig(compression=CompressionRatio.X16, always_ram=True)
            )
        )


def _upsert_chunk_with_retry(db, chunk: str, metadata: dict, item_id: str, mode: str) -> None:
    vreme_cekanja = 2
    uspesno_indeksirano = False

    while not uspesno_indeksirano:
        try:
            db.add_texts(
                texts=[chunk],
                metadatas=[metadata],
                ids=[item_id]
            )
            uspesno_indeksirano = True
            
            if mode == "Cloud":
                time.sleep(0.5)

        except Exception as e:
            if mode == "Cloud" and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                print(f"      ⚠️ API limit dostignut. Čekam {vreme_cekanja}s pre novog pokušaja...")
                time.sleep(vreme_cekanja)
                vreme_cekanja *= 2
            else:
                print(f"   ❌ Kritična greška tokom upisa u [{mode}]: {e}")
                raise e


def populate_vector_store(chunks: list[str], mode: str) -> None:
    """Upravlja kompletnim procesom pripreme i punjenja za jednu specifičnu bazu."""
    print(f"\n" + "="*50)
    print(f" POKREĆEM UPIS PODATAKA U [{mode.upper()}] BAZU...")
    print("="*50)


    st.session_state["rezim_rada"] = mode
    
    from config.vector_stores import get_vector_store
    db = get_vector_store()

    if mode == "Local":
        _init_qdrant_collection(db)

    ukupan_broj = len(chunks)
    for index_num, chunk in enumerate(chunks):
        item_id = str(index_num + 1)
        metadata = {
            "text": chunk,
            "category": CATEGORY_NAME,
            "item_number": index_num + 1
        }

        print(f"   [{mode}] Upisujem chunk {item_id}/{ukupan_broj}")
        _upsert_chunk_with_retry(db, chunk, metadata, item_id, mode)

    print(f"✅ [Uspeh] Upis podataka u [{mode}] bazu je kompletno završen!")


def main():
    if not os.path.exists(FILE_PATH):
        print(f" Greška: Fajl '{FILE_PATH}' ne postoji.")
        return

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f" Učitavam fajl '{FILE_PATH}'...")
    chunks = chunk_document_by_delimiters(raw_text)
    print(f" Tekst je uspešno podeljen na {len(chunks)} celina.")

    REŽIMI_RADA = ["Local", "Cloud"]
    
    for aktuelni_mod in REŽIMI_RADA:
        populate_vector_store(chunks, aktuelni_mod)

    print(f"\n KRAJ! Obe baze (Local i Cloud) su uspešno sinhronizovane i napunjene podacima!")


if __name__ == "__main__":
    main()