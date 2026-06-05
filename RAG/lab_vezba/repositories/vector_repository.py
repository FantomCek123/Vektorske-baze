import streamlit as st
from config.vector_stores import get_vector_store, get_qdrant_client, get_pinecone_client, COLLECTION_NAME

def _get_db(mode: str = None):
    db = get_vector_store(mode=mode)
    db.fetch_native = lambda _id: db.get_by_ids([_id])[0].metadata if db.get_by_ids([_id]) else None
    db.delete_native = lambda _id: db.delete(ids=[_id])
    return db


def is_db_empty(mode: str) -> bool:
    try:
        if mode == "Local":
            client = get_qdrant_client()
            if not client.collection_exists(COLLECTION_NAME):
                return True
            
            collection_info = client.get_collection(COLLECTION_NAME)
            count = getattr(collection_info, 'points_count', 0)
            return count == 0
            
        else: # Cloud 
            pc = get_pinecone_client()
            if COLLECTION_NAME not in [idx['name'] for idx in pc.list_indexes()]:
                return True
            
            stats = pc.Index(COLLECTION_NAME).describe_index_stats()
            return stats.total_vector_count == 0
            
    except Exception as e:
        # Logovanje greške nam pomaže da ne nagađamo šta nije u redu
        print(f"DEBUG: Greska pri proveri statusa baze: {e}", flush=True)
        return True

def create_or_update_scene(_id: str, tekst: str, film: str, mode: str = None) -> bool:
    try:
        db = _get_db(mode=mode)
        db.add_texts(texts=[tekst], metadatas=[{"text": tekst, "film": film, "scena_broj": int(_id) if _id.isdigit() else _id}], ids=[_id])
        return True
    except Exception as e:
        print(f"Upsert Greška: {e}")
        return False

def read_scene(_id: str, mode: str = None):
    return _get_db(mode=mode).fetch_native(_id)

def delete_scene(_id: str, mode: str = None) -> bool:
    try:
        _get_db(mode=mode).delete_native(_id)
        return True
    except Exception as e:
        print(f"Delete Greška: {e}")
        return False