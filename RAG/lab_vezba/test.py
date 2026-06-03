import google.generativeai as genai
import os

def check_available_models():
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    for m in genai.list_models():
        if 'embed' in m.name:
            print(f"Dostupan model: {m.name}")