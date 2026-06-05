import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def get_embeddings(rezim_rada):
    if rezim_rada == "Local":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )

@st.cache_resource
def get_llm(rezim_rada):
    if rezim_rada == "Local":
        model_id = "microsoft/phi-1_5" 
        print(1,flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(2,flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            attn_implementation="eager",
            device_map="cpu",
            torch_dtype=torch.float32, 
            trust_remote_code=True,
            low_cpu_mem_usage=True    
        )
        print(3,flush=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
        print(4,flush=True)
        return HuggingFacePipeline(pipeline=pipe)
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.1
        )