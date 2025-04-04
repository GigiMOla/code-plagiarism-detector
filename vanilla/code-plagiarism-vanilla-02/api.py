from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import faiss
import pickle
import torch
import re
import os
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from config.settings import *
from typing import Optional

app = FastAPI(title="Multi-Language Plagiarism Detector")

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize services
client = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Load FAISS index and metadata
index = faiss.read_index(os.path.join(INDEX_DIR, 'embeddings.faiss'))
with open(os.path.join(INDEX_DIR, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

class CodeRequest(BaseModel):
    code: str
    language: Optional[str] = 'python'

def normalize_identifiers(code, language='python'):
    """Normalize variable and function names"""
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
    """Standardized code preprocessing"""
    config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG['python'])
    
    # Remove comments and type hints
    for pattern in config['comment_patterns']:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.MULTILINE)
    for pattern in config['type_hint_patterns']:
        content = re.sub(pattern, '', content)
    
    # Normalization steps
    content = re.sub(r'"[^"]*"', '"str"', content)
    content = re.sub(r"'[^']*'", "'str'", content)
    content = re.sub(r'\b\d+\b', '0', content)
    content = normalize_identifiers(content, language)
    content = re.sub(r'\b(len|range|str|int|float|print)\b', 'std_func', content)
    
    # Filter and trim lines
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    filtered_lines = [
        line for line in lines
        if not line.startswith(('import ', 'from ', '#include ', 'package ', 'using '))
    ]
    
    return '\n'.join(filtered_lines[:PROCESSING_CONFIG['max_code_lines']])

def get_embedding(text, language='python'):
    """Generate embedding with proper preprocessing"""
    processed = preprocess_code(text, language)
    inputs = tokenizer(
        processed,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Attention-weighted pooling
    attention_mask = inputs['attention_mask']
    last_hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
    embedding = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    
    # Normalize and return
    embedding = embedding.squeeze().numpy()
    return (embedding / np.linalg.norm(embedding)).astype('float32')

def is_plagiarized(query_code: str, matches: list) -> bool:
    """Enhanced LLM analysis with strict parsing"""
    system_msg = """Analyze if the query code is copied from matches. Consider:
    1. Algorithm implementation
    2. Variable/function patterns
    3. Control flow
    4. Unique idioms
    
    Return ONLY 'true' or 'false'."""
    
    matches_text = []
    for i, match in enumerate(matches[:3]):
        code_lines = match['code'].split('\n')[:30]
        matches_text.append(
            f"=== Match {i+1} (Similarity: {match['similarity']:.2f}) ===\n"
            f"File: {match['file']}\n"
            f"Code:\n{'\n'.join(code_lines)}\n"
        )

    prompt = f"""Code Analysis Request:
    
[Query Code]
{query_code}

[Matches]
{'#' * 50}
{'\n'.join(matches_text)}
{'#' * 50}

Is the query substantially similar to any match? (true/false)"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=5
        )
        
        verdict = response.choices[0].message.content.strip().lower()
        if re.search(r'\btrue\b', verdict):
            return True
        elif re.search(r'\bfalse\b', verdict):
            return False
        return any(m['similarity'] >= 0.85 for m in matches)
    
    except Exception as e:
        print(f"LLM Error: {e}")
        return any(m['similarity'] >= 0.85 for m in matches)

@app.on_event("startup")
async def startup_checks():
    """Verify required files exist"""
    required = [
        os.path.join(INDEX_DIR, 'embeddings.faiss'),
        os.path.join(INDEX_DIR, 'metadata.pkl')
    ]
    for path in required:
        if not os.path.exists(path):
            raise RuntimeError(f"Missing index file: {path}. Run index.py first!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "languages": list(LANGUAGE_CONFIG.keys())
    })

@app.post("/check", response_class=HTMLResponse)
async def check_plagiarism(
    request: Request,
    code: str = Form(...),
    language: str = Form('python')
):
    try:
        # Validate input
        processed = preprocess_code(code, language)
        if len(processed.split('\n')) < PROCESSING_CONFIG['min_code_lines']:
            raise ValueError(
                f"Code too short ({len(processed.split('\n'))} lines). "
                f"Minimum required: {PROCESSING_CONFIG['min_code_lines']}"
            )

        # Search for matches
        embedding = get_embedding(code, language).reshape(1, -1)
        similarities, indices = index.search(embedding, TOP_K)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if metadata[idx]["language"] == language:
                with open(metadata[idx]["path"], 'r', encoding='utf-8') as f:
                    similar_code = f.read()
                results.append({
                    "file": metadata[idx]["path"],
                    "similarity": float(sim),
                    "code": similar_code
                })
        
        # Determine verdict
        top_sim = max([m['similarity'] for m in results]) if results else 0.0
        plagiarized = is_plagiarized(code, results) if results else False
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "input_code": code,
            "language": language,
            "results": results,
            "plagiarized": plagiarized,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "top_similarity": top_sim
        })
    
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.post("/api/check")
async def api_check(request: CodeRequest):
    try:
        processed = preprocess_code(request.code, request.language)
        if len(processed.split('\n')) < PROCESSING_CONFIG['min_code_lines']:
            raise HTTPException(400, detail="Code too short after preprocessing")
        
        embedding = get_embedding(request.code, request.language).reshape(1, -1)
        similarities, indices = index.search(embedding, TOP_K)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if metadata[idx]["language"] == request.language:
                with open(metadata[idx]["path"], 'r', encoding='utf-8') as f:
                    similar_code = f.read()
                results.append({
                    "file": metadata[idx]["path"],
                    "similarity": float(sim),
                    "code": similar_code
                })
        
        return {
            "plagiarized": is_plagiarized(request.code, results),
            "matches": len(results),
            "top_similarity": max([m['similarity'] for m in results]) if results else 0.0
        }
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))