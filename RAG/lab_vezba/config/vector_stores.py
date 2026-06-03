import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_pinecone import PineconeVectorStore
from config.models import get_embeddings

def _ensure_qdrant_collection(client, collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

def _ensure_pinecone_index(index_name):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

def get_vector_store():
    rezim_rada = st.session_state.get("rezim_rada", "Local")
    embeddings = get_embeddings(rezim_rada)
    collection_name = "taxi-driver-screenplay"
    
    if rezim_rada == "Local":
        client = QdrantClient(url="http://qdrant-db:6333")
        _ensure_qdrant_collection(client, collection_name)
        db = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
    else:
        _ensure_pinecone_index(collection_name)
        db = PineconeVectorStore(
            index_name=collection_name,
            embedding=embeddings,
            pinecone_api_key=os.environ.get("PINECONE_API_KEY")
        )
        
    db.fetch_native = lambda _id: db.get_by_ids([_id])[0].metadata if db.get_by_ids([_id]) else None
    db.delete_native = lambda _id: db.delete(ids=[_id])
        
    return db