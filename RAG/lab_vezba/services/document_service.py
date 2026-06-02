import re

def chunk_document_by_delimiters(text: str) -> list[str]:
    """
    Parsira sirovi tekst i deli ga na logičke celine na osnovu prepoznatih markera.
    Agnostička funkcija primenjiva na scenarije, knjige ili dokumentaciju.
    """
    raw_chunks = re.split(r'(CUT TO:|INT\.|EXT\.)', text)
    chunks = []
    current_chunk = ""
    
    for part in raw_chunks:
        if part in ["CUT TO:", "INT.", "EXT."]:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = part + " "
        else:
            current_chunk += part
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 15]