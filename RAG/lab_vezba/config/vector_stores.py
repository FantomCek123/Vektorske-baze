import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_pinecone import PineconeVectorStore
from config.models import get_embeddings

COLLECTION_NAME = "taxi-driver-screenplay"

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url="http://qdrant-db:6333")

@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def _ensure_qdrant_collection(client, collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

def _ensure_pinecone_index(pc, index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=3072, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

def get_vector_store(mode=None):
    mode = mode if mode else st.session_state.get("rezim_rada", "Local")
    embeddings = get_embeddings(mode)
    
    if mode == "Local":
        client = get_qdrant_client()
        _ensure_qdrant_collection(client, COLLECTION_NAME)
        return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    else:
        pc = get_pinecone_client()
        _ensure_pinecone_index(pc, COLLECTION_NAME)
        return PineconeVectorStore(
            index_name=COLLECTION_NAME, 
            embedding=embeddings, 
            pinecone_api_key=os.environ.get("PINECONE_API_KEY")
        )