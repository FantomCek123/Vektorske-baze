import os
import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_pinecone import PineconeVectorStore
from config.models import get_embeddings

def get_vector_store():
    rezim_rada = st.session_state.get("rezim_rada", "Local")
    embeddings = get_embeddings()
    
    if rezim_rada == "Local":
        client = QdrantClient(url="http://qdrant_local:6333")
        db = QdrantVectorStore(
            client=client,
            collection_name="taxi_driver_screenplay",
            embedding=embeddings
        )
    else:
        db = PineconeVectorStore(
            index_name="lab-vezba-cloud",
            embedding=embeddings,
            pinecone_api_key=os.environ.get("PINECONE_API_KEY")
        )
        
    db.fetch_native = lambda _id: db.get_by_ids([_id])[0].metadata if db.get_by_ids([_id]) else None
    db.delete_native = lambda _id: db.delete(ids=[_id])
        
    return db