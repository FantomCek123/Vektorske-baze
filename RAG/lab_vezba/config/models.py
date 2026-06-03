import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

@st.cache_resource
def get_embeddings(rezim_rada):
    if rezim_rada == "Local":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

@st.cache_resource
def get_llm(rezim_rada):
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
        # gpt-4o-mini je brz, jeftin i izuzetno pametan
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.1
        )