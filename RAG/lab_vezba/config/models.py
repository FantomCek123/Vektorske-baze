import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def get_embeddings():
    rezim_rada = st.session_state.get("rezim_rada", "Local")
    if rezim_rada == "Local":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            output_dimensionality=768,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )

def get_llm():
    rezim_rada = st.session_state.get("rezim_rada", "Local")
    
    if rezim_rada == "Local":
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            trust_remote_code=True
        )
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=100
        )
        
        return HuggingFacePipeline(pipeline=pipe)
        
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.1
        )