import os
import uuid
import logging
import streamlit as st
from dotenv import load_dotenv
from services.document_service import chunk_document_by_delimiters
from repositories.vector_repository import is_db_empty, _get_db, create_or_update_scene

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



load_dotenv()
FILE_PATH = "Taxi-Driver-Script.txt"

def run_ingestion(chunks: list, mode: str):
    logger.info(f"Initializing {mode} pipeline")
    st.session_state["rezim_rada"] = mode
    
    if not is_db_empty(mode):
        logger.info(f"Podaci već postoje u {mode} bazi. Preskačem ingestiju.")
        return

    logger.info(f"Baza je prazna. Pokrećem ingestiju...")
    db = _get_db(mode)
    _index_chunks(db, chunks, mode)
    logger.info(f"Pipeline {mode} finished")


def _upsert_batch(db, batch_texts, batch_metas, batch_ids, mode, max_retries=5):
    attempt = 0
    backoff = 2
    
    while attempt < max_retries:
        try:
            db.add_texts(texts=batch_texts, metadatas=batch_metas, ids=batch_ids)
            return  
            
        except Exception as e:
            error_str = str(e)
            is_rate_limit = mode == "Cloud" and any(code in error_str for code in ["429", "RESOURCE_EXHAUSTED"])
            
            if not is_rate_limit:
                logger.error(f"Kritična greška (ne može se retry-ovati): {e}")
                raise e
            
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"Dostignut maksimalan broj pokušaja ({max_retries}). Odustajem.")
                raise e
                
            logger.warning(f"Rate limit (pokušaj {attempt}/{max_retries}). Čekam {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60) 

def _index_chunks(db, chunks: list, mode: str):
    total = len(chunks)
    BATCH_SIZE = 1 if mode == "Local" else 20
    for i in range(0, total, BATCH_SIZE):
        batch_texts = chunks[i : i + BATCH_SIZE]
        batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_texts))]
        batch_metas = [{"text": t, "category": "Taxi Driver"} for t in batch_texts]
        
        logger.info(f"Indexing [{mode}] batch {i // BATCH_SIZE + 1}/{(total // BATCH_SIZE) + 1}")
        _upsert_batch(db, batch_texts, batch_metas, batch_ids, mode)


def main():
    try:
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"File {FILE_PATH} not found")
            
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            chunks = chunk_document_by_delimiters(f.read())
            
        for mode in ["Local", "Cloud"]:
            run_ingestion(chunks, mode)
    except Exception as e:
        logger.critical(f"Pipeline failure: {e}")

if __name__ == "__main__":
    main()