import os
import time
import logging
from dotenv import load_dotenv
from services.document_service import chunk_document_by_delimiters
from config.vector_stores import get_vector_store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

FILE_PATH = "Taxi-Driver-Script.txt"
CATEGORY = "Taxi Driver"
BATCH_SIZE = 20

def _upsert_batch(db, batch_texts, batch_metas, batch_ids, mode):
    backoff = 2
    while True:
        try:
            db.add_texts(texts=batch_texts, metadatas=batch_metas, ids=batch_ids)

        except Exception as e:
            if mode == "Cloud" and any(code in str(e) for code in ["429", "RESOURCE_EXHAUSTED"]):
                logger.warning(f"Rate limit hit. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                logger.error(f"Batch ingestion failed for {mode}: {e}")
                raise e

def _index_chunks(db, chunks: list, mode: str):
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch_texts = chunks[i : i + BATCH_SIZE]
        batch_ids = [str(j + 1) for j in range(i, i + len(batch_texts))]
        batch_metas = [{"text": t, "category": CATEGORY, "item_number": id} for t, id in zip(batch_texts, batch_ids)]
        
        logger.info(f"Indexing [{mode}] batch {i // BATCH_SIZE + 1}/{(total // BATCH_SIZE) + 1}")
        _upsert_batch(db, batch_texts, batch_metas, batch_ids, mode)

def run_ingestion(chunks: list, mode: str):
    logger.info(f"Initializing {mode} pipeline")
    import streamlit as st
    st.session_state["rezim_rada"] = mode
    
    db = get_vector_store()
    _index_chunks(db, chunks, mode)
    logger.info(f"Pipeline {mode} finished")

def main():
    try:
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"File {FILE_PATH} not found")
            
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            chunks = chunk_document_by_delimiters(f.read())
            
        logger.info(f"Data ready. Total segments: {len(chunks)}")
        for mode in ["Cloud", "Local"]:
            run_ingestion(chunks, mode)
    except Exception as e:
        logger.critical(f"Pipeline failure: {e}")

if __name__ == "__main__":
    main()