import streamlit as st
from config.vector_stores import get_vector_store

def create_or_update_scene(_id: str, tekst: str, film: str) -> bool:
    try:
        db = get_vector_store()
        db.add_texts(
            texts=[tekst],
            metadatas=[{"text": tekst, "film": film, "scena_broj": int(_id) if _id.isdigit() else _id}],
            ids=[_id]
        )
        return True
    except Exception as e:
        print(f"Upsert Greška: {e}")
        return False

def read_scene(_id: str):
    try:
        db = get_vector_store()
        return db.fetch_native(_id)
    except Exception as e:
        print(f"Read Greška: {e}")
    return None

def delete_scene(_id: str) -> bool:
    try:
        db = get_vector_store()
        db.delete_native(_id)
        return True
    except Exception as e:
        print(f"Delete Greška: {e}")
        return False