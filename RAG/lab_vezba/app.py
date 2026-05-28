import streamlit as st
from dotenv import load_dotenv
load_dotenv()

modeli_spremni = "izvrseno_ucitavanje" in st.session_state

if modeli_spremni:
    from local.local_rag import ask_local_rag
    from cloud.cloud_rag import ask_cloud_rag

with st.sidebar:
    st.title("Podešavanja")
    rezim_rada = st.radio("Izaberi model pretrage:", ("Local", "Cloud"))
    st.markdown("---")
    placeholder_stats = st.empty()

left_pad, main_col, right_pad = st.columns([1, 3, 1])

with main_col:
    st.title("Taxi Driver chatbot")
    st.markdown("---")
    input_placeholder = st.container()

with input_placeholder:
    user_pitanje = st.chat_input("Postavi pitanje...")

if user_pitanje and modeli_spremni:
    with main_col:
        with st.chat_message("user"):
            st.write(user_pitanje)
            
        with st.chat_message("assistant"):
            with st.spinner("AI analizira..."):
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

if not modeli_spremni:
    with main_col:
        with st.spinner("Ucitavanje..."):
            from local.local_rag import init_local_clients
            from cloud.cloud_rag import init_cloud_clients
            
            init_local_clients()
            init_cloud_clients()
            
            st.session_state["izvrseno_ucitavanje"] = True
            st.rerun()