import streamlit as st
from dotenv import load_dotenv

from local.local_rag import ask_local_rag
from cloud.cloud_rag import ask_cloud_rag

load_dotenv()

st.set_page_config(page_title="Filmski RAG Asistent", layout="wide")

with st.sidebar:
    st.title("Podešavanja")
    rezim_rada = st.radio(
        "Izaberi model pretrage:",
        ("Local", "Cloud"),
    )
    st.markdown("---")
    placeholder_stats = st.empty()

left_pad, main_col, right_pad = st.columns([1, 2, 1])

with main_col:
    st.title("Taxi Driver chatbot")
    st.markdown("---")
    input_placeholder = st.container()

with input_placeholder:
    user_pitanje = st.chat_input("Postavi pitanje...")

if user_pitanje:
    with main_col:
        with st.chat_message("user"):
            st.write(user_pitanje)
            
        with st.chat_message("assistant"):
            with st.spinner("AI analizira..."):
                # Pozivamo funkcije koje smo uvezli sa vrha
                if rezim_rada == "Local":
                    odgovor, t_search, t_gen = ask_local_rag(user_pitanje)
                else:
                    odgovor, t_search, t_gen = ask_cloud_rag(user_pitanje)
                st.write(odgovor)
        
    with placeholder_stats.container():
        st.markdown("### Statistika poslednjeg upita")
        st.metric("Pretraga (Vektori)", f"{t_search:.3f} s")
        st.metric("Generisanje (Model)", f"{t_gen:.3f} s")
        st.metric("Ukupno", f"{t_search + t_gen:.3f} s")

st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        left: 0 !important;
        right: 0 !important;
        margin: auto !important;
        width: 50% !important; 
    }
    </style>
""", unsafe_allow_html=True)