import streamlit as st
from dotenv import load_dotenv
import os
import warnings  
import logging   
load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*flash-attention.*")
warnings.filterwarnings("ignore", message=".*Special tokens have been added.*")


if "rezim_rada" not in st.session_state:
    st.session_state["rezim_rada"] = "Local"

modeli_spremni = "izvrseno_ucitavanje" in st.session_state

if modeli_spremni:
    from services.rag_service import ask_rag
    from repositories.vector_repository import create_or_update_scene, read_scene, delete_scene

with st.sidebar:
    st.title("Podešavanja")
    

    st.radio("Izaberi model pretrage:", ("Local", "Cloud"), key="rezim_rada")    
    st.markdown("---")
    
    tip_stranice = st.selectbox("Izaberi režim rada:", ["Chatbot", "CRUD"])
    st.markdown("---")
    
    placeholder_stats = st.empty()

rezim_rada_trenutni = st.session_state["rezim_rada"]

left_pad, main_col, right_pad = st.columns([1, 3, 1])

if tip_stranice == "Chatbot":
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
                    odgovor, t_search, t_gen = ask_rag(user_pitanje)
                    st.write(odgovor)
            
        with placeholder_stats.container():
            st.markdown("### Statistika poslednjeg upita")
            st.metric("Pretraga (Vektori)", f"{t_search:.3f} s")
            st.metric("Generisanje (Model)", f"{t_gen:.3f} s")
            st.metric("Ukupno", f"{t_search + t_gen:.3f} s")

elif tip_stranice == "CRUD" and modeli_spremni:
    with main_col:        
        scena_id = st.number_input("ID Scene:", min_value=1)
        crud_akcija = st.radio("", ["CREATE / UPDATE", "READ", "DELETE"], horizontal=True)

        if crud_akcija == "CREATE / UPDATE":
            tekst_scene = st.text_area("Tekst scene / kontekst:")
            film_ime = st.text_input("Naziv filma:", value="Taxi Driver")
            
            if st.button("Izvrši upis u bazu"):
                if tekst_scene:
                    with st.spinner("Upisivanje..."):
                        uspeh = create_or_update_scene(scena_id, tekst_scene, film_ime, mode=rezim_rada_trenutni)
                    
                    if uspeh:
                        st.success(f"Uspešno upisano u {rezim_rada_trenutni} bazu!")
                else:
                    st.error("Tekst scene ne može biti prazan!")
                        
        elif crud_akcija == "READ":
            if st.button("Pronađi scenu"):
                with st.spinner("Pretraga..."):
                    podaci = read_scene(str(scena_id), mode=rezim_rada_trenutni)
                        
                if podaci:
                    st.json(podaci)
                else:
                    st.warning(f"Nema podataka za ovaj ID.")
                    
        elif crud_akcija == "DELETE":
            potvrda = st.checkbox("Dodatna potvrda")
            if st.button("Obriši iz baze"):
                if potvrda:
                    with st.spinner("Brisanje..."):
                        uspeh = delete_scene(str(scena_id), mode=rezim_rada_trenutni)
                    if uspeh:
                        st.success("Obrisano!")
                else:
                    st.error("Moraš štiklirati potvrdu!")

if not modeli_spremni:
    with main_col:
        st.subheader("Inicijalizacija sistema")
        progres_bar = st.progress(0)
        
        with st.spinner("Učitavanje svih servisa..."):
            from repositories.vector_repository import _get_db
            from config.models import get_llm
            
            st.session_state["db_local"] = _get_db(mode="Local")
            progres_bar.progress(25)        
            st.session_state["llm_local"] = get_llm("Local")
            progres_bar.progress(50)
            
            st.session_state["db_cloud"] = _get_db(mode="Cloud")
            progres_bar.progress(75)
            st.session_state["llm_cloud"] = get_llm("Cloud")
            progres_bar.progress(100)
            
            st.session_state["izvrseno_ucitavanje"] = True
            st.rerun()