import streamlit as st
from dotenv import load_dotenv
load_dotenv()

modeli_spremni = "izvrseno_ucitavanje" in st.session_state

if modeli_spremni:
    from local.local_rag import ask_local_rag
    from cloud.cloud_rag import ask_cloud_rag
    import CRUD.qdrant_crud as qd
    import CRUD.pinecone_crud as pc

with st.sidebar:
    st.title("Podešavanja")
    rezim_rada = st.radio("Izaberi model pretrage:", ("Local", "Cloud"))
    st.markdown("---")
    
    tip_stranice = st.selectbox("Izaberi režim rada:", ["Chatbot", "CRUD"])
    st.markdown("---")
    
    placeholder_stats = st.empty()

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

elif tip_stranice == "CRUD" and modeli_spremni:
    with main_col:        
        scena_id = st.number_input("ID Scene:", min_value=1)

        crud_akcija = st.radio("",["CREATE / UPDATE", "READ", "DELETE"], horizontal=True)

        
        if crud_akcija == "CREATE / UPDATE":
            tekst_scene = st.text_area("Tekst scene / kontekst:")
            if st.button("Izvrši upis u bazu"):
                if tekst_scene:
                    with st.spinner("Upisivanje i generisanje embeddinga..."):
                        if rezim_rada == "Local":
                            uspeh = qd.create_or_update_qdrant(scena_id, tekst_scene, film_ime)
                        else:
                            uspeh = pc.create_or_update_pinecone(scena_id, tekst_scene, film_ime)
                    
                    if uspeh:
                        st.success(f"Uspešno upisano u {rezim_rada} bazu pod ID-jem: {scena_id if rezim_rada == 'Local' else f'scena_{scena_id}'}")
                    else:
                        st.error("Baza je vratila grešku prilikom upisa.")
                else:
                    st.error("Tekst scene ne može biti prazan!")
                    
        elif crud_akcija == "READ":
            if st.button("Pronađi scenu"):
                with st.spinner("Pretraga po ID-ju..."):
                    if rezim_rada == "Local":
                        podaci = qd.read_qdrant(scena_id)
                    else:
                        podaci = pc.read_pinecone(scena_id)
                        
                if podaci:
                    st.markdown(f"**Pronađeni podaci u {rezim_rada} bazi:**")
                    st.json(podaci)
                else:
                    st.warning(f"Nema podataka za ovaj ID u {rezim_rada} bazi.")
                    
        elif crud_akcija == "DELETE":
            potvrda = st.checkbox("Dodatna potvrda")
            
            if st.button("Obriši iz baze"):
                if potvrda:
                    with st.spinner("Brisanje iz baze..."):
                        if rezim_rada == "Local":
                            uspeh = qd.delete_qdrant(scena_id)
                        else:
                            uspeh = pc.delete_pinecone(scena_id)
                            
                    if uspeh:
                        st.success(f"Scena uspešno obrisana iz {rezim_rada} baze.")
                    else:
                        st.error("Došlo je do greške prilikom brisanja.")
                else:
                    st.error("Moraš prvo štiklirati polje za potvrdu brisanja!")

# ==================== INITIAL LOADING SYSTEM ====================
if not modeli_spremni:
    with main_col:
        st.subheader("Inicijalizacija sistema")

        
        
        with st.spinner("Lokalni modeli"):
            progres_bar = st.progress(0)
            from local.local_rag import init_local_clients
            init_local_clients()
        progres_bar.progress(25)
        
        with st.spinner("Cloud servisi"):
            from cloud.cloud_rag import init_cloud_clients
            init_cloud_clients()
        progres_bar.progress(50)
        
        with st.spinner("Qdrant moduli"):
            import CRUD.qdrant_crud as qd
        progres_bar.progress(75)
        
        with st.spinner("Pinecone moduli"):
            import CRUD.pinecone_crud as pc
        progres_bar.progress(100)
        
        # Postavljanje stanja i automatski rerun aplikacije
        st.session_state["izvrseno_ucitavanje"] = True
        st.rerun()