import time
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.models import get_llm
from config.vector_stores import get_vector_store

def _format_docs(docs):
    if not docs:
        return "Nema pronađenog konteksta."
    return "\n\n".join([doc.page_content for doc in docs])

def ask_rag(user_query: str):
    rezim_rada = st.session_state.get("rezim_rada", "Local")
    

    llm = st.session_state["llm_local"] if st.session_state["rezim_rada"] == "Local" else st.session_state["llm_cloud"]
    db = st.session_state["db_local"] if st.session_state["rezim_rada"] == "Local" else st.session_state["db_cloud"]
    
    # ... dalje tvoja RAG logika
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = ChatPromptTemplate.from_template(
        "Odgovori kratko na srpskom jeziku. Kontekst: {context} Pitanje: {question} u filmu Taxi Driver"
    )
    
    start_pretraga = time.time()
    pronadjeni_dokumenti = retriever.invoke(user_query)
    vreme_pretrage = time.time() - start_pretraga
    
    cist_kontekst = _format_docs(pronadjeni_dokumenti)
    
    uformisan_prompt = prompt_template.format_messages(
        context=cist_kontekst,
        question=user_query
    )
    rag_chain = llm | StrOutputParser()
    
    start_generisanje = time.time()
    try:
        odgovor = rag_chain.invoke(uformisan_prompt)
        vreme_generisanja = time.time() - start_generisanje
        
        if rezim_rada == "Cloud" and not odgovor:
            odgovor = "Žao mi je, ali ne mogu da odgovorim na ovo pitanje zbog sigurnosnih filtera."
            
    except Exception as e:
        return f"Greška pri [{rezim_rada}] RAG generisanju: {e}", vreme_pretrage, 0
        
    return odgovor, vreme_pretrage, vreme_generisanja