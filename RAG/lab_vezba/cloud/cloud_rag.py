import os
import time
import streamlit as st
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

INDEX_NAME = "lab-vezba-cloud"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

@st.cache_resource
def init_cloud_clients():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_client = pc.Index(INDEX_NAME)
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768,
        google_api_key=GEMINI_API_KEY
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )
    return index_client, embedding_model, llm

def ask_cloud_rag(user_query):
    index_client, embedding_cloud, llm = init_cloud_clients()
    start_pretraga = time.time()
    
    try:
        query_vector = embedding_cloud.embed_query(user_query)
    except Exception as e:
        return f"Greška pri generisanju cloud vektora: {e}", 0, 0

    try:
        search_results = index_client.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
    except Exception as e:
        return f"Greška pri pretrazi Pinecone baze: {e}", 0, 0
    
    vreme_pretrage = time.time() - start_pretraga
    context = "Nema pronađenog konteksta." if not search_results.matches else search_results.matches[0].metadata["text"]
    prompt = f"Odgovori kratko na srpskom jeziku. Kontekst: {context} Pitanje: {user_query}"

    start_generisanje = time.time()
    try:
        response = llm.invoke(prompt)
        vreme_generisanja = time.time() - start_generisanje
        if not response.content:
            odgovor = "Žao mi je, ali ne mogu da odgovorim na ovo pitanje zbog sigurnosnih filtera."
        elif isinstance(response.content, list):
            if len(response.content) > 0 and isinstance(response.content[0], dict):
                odgovor = response.content[0].get("text", "Nema teksta u odgovoru.")
            else:
                odgovor = str(response.content)
        else:
            odgovor = response.content
    except Exception as e:
        return f"Greška pri generisanju preko Gemini API: {e}", vreme_pretrage, 0

    return odgovor, vreme_pretrage, vreme_generisanja