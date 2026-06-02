import streamlit as st
from dotenv import load_dotenv
load_dotenv()


if "rezim_rada" not in st.session_state:
    st.session_state["rezim_rada"] = "Local"

modeli_spremni = "izvrseno_ucitavanje" in st.session_state

if modeli_spremni:
    from api.rag_service import ask_rag
    from api.crud_service import create_or_update_scene, read_scene, delete_scene

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
                    with st.spinner("Upisivanje i generisanje embeddinga..."):
                        uspeh = create_or_update_scene(scena_id, tekst_scene, film_ime)
                    
                    if uspeh:
                        st.success(f"Uspešno upisano u {rezim_rada_trenutni} bazu pod ID-jem: {scena_id if rezim_rada_trenutni == 'Local' else f'scena_{scena_id}'}")
                    else:
                        st.error("Baza je vratila grešku prilikom upisa.")
                else:
                    st.error("Tekst scene ne može biti prazan!")
                    
        elif crud_akcija == "READ":
            if st.button("Pronađi scenu"):
                with st.spinner("Pretraga po ID-ju..."):
                    podaci = read_scene(scena_id)
                        
                if podaci:
                    st.markdown(f"**Pronađeni podaci u {rezim_rada_trenutni} bazi:**")
                    st.json(podaci)
                else:
                    st.warning(f"Nema podataka za ovaj ID u {rezim_rada_trenutni} bazi.")
                    
        elif crud_akcija == "DELETE":
            potvrda = st.checkbox("Dodatna potvrda")
            
            if st.button("Obriši iz baze"):
                if potvrda:
                    with st.spinner("Brisanje iz baze..."):
                        uspeh = delete_scene(scena_id)
                            
                    if uspeh:
                        st.success(f"Scena uspešno obrisana iz {rezim_rada_trenutni} baze.")
                    else:
                        st.error("Došlo je do greške prilikom brisanja.")
                else:
                    st.error("Moraš prvo štiklirati polje za potvrdu brisanja!")

if not modeli_spremni:
    with main_col:
        st.subheader("Inicijalizacija sistema")
        progres_bar = st.progress(0)
        
        from config.vector_stores import get_vector_store
        from config.models import get_llm

        with st.spinner("Učitavanje lokalnih modela i Qdrant-a..."):
            st.session_state["rezim_rada"] = "Local"
            get_vector_store()
            get_llm()
            progres_bar.progress(50)
        
        with st.spinner("Učitavanje cloud servisa i Pinecone-a..."):
            st.session_state["rezim_rada"] = "Cloud"
            get_vector_store()
            get_llm()
            progres_bar.progress(100)
        
        st.session_state["rezim_rada"] = "Local"
        st.session_state["izvrseno_ucitavanje"] = True
        st.rerun()