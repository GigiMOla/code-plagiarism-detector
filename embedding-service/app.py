from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
from config.settings import LANGUAGE_CONFIG, PROCESSING_CONFIG
import re

app = FastAPI()

class EmbeddingRequest(BaseModel):
    text: str
    language: str = "python"

# Initialize model
tokenizer = None
model = None
device = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def normalize_identifiers(code, language='python'):
    """Your existing normalization function"""
    config = LANGUAGE_CONFIG.get(language, {})
    preserved_vars = {'i', 'j', 'k', 'n', 'm', 'x', 'y', 'z', 'temp', 'swap'}
    keywords = config.get('keywords', set())

    var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'

    def replace_match(match):
        word = match.group(1)
        if word in preserved_vars or word in keywords:
            return word
        if word.isupper():
            return 'CONST'
        if word[0].isupper():
            return 'Class'
        return 'var'

    return re.sub(var_pattern, replace_match, code)

def preprocess_code(content, language='python'):
    """Your existing preprocessing function"""
    config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG['python'])

    for pattern in config['comment_patterns']:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.MULTILINE)
    for pattern in config['type_hint_patterns']:
        content = re.sub(pattern, '', content)

    content = re.sub(r'"[^"]*"', '"str"', content)
    content = re.sub(r"'[^']*'", "'str'", content)
    content = re.sub(r'\b\d+\b', '0', content)
    content = normalize_identifiers(content, language)
    content = re.sub(r'\b(len|range|str|int|float|print)\b', 'std_func', content)

    lines = [line.strip() for line in content.split('\n') if line.strip()]
    filtered_lines = [
        line for line in lines
        if not line.startswith(('import ', 'from ', '#include ', 'package ', 'using '))
    ]

    return '\n'.join(filtered_lines[:PROCESSING_CONFIG['max_code_lines']])

@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    try:
        processed = preprocess_code(request.text, request.language)
        inputs = tokenizer(
            processed,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        attention_mask = inputs["attention_mask"]
        last_hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        embedding = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(500, detail=str(e))