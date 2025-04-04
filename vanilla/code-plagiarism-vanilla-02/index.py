import os
import subprocess
import torch
import numpy as np
import faiss
import pickle
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from config.settings import *

def clone_repositories():
    """Clone all repositories with error handling and progress tracking"""
    print("Checking repositories...")
    with open(REPO_FILE) as f:
        repos = [r.strip() for r in f.readlines() if r.strip()]
    
    os.makedirs(REPO_DIR, exist_ok=True)
    
    for repo in repos:
        if not repo or not repo.startswith(('http', 'git@')):
            continue
            
        repo_name = repo.split('/')[-1].replace('.git', '')
        target_dir = os.path.join(REPO_DIR, repo_name)
        
        if os.path.exists(target_dir):
            print(f"Repository {repo_name} already exists, skipping...")
            continue
            
        print(f"Cloning {repo_name}...")
        try:
            subprocess.run(
                ['git', 'clone', '--depth', '1', repo, target_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo}: {e.stderr.decode().strip()}")
            continue

def normalize_identifiers(code, language='python'):
    """Consistent with api.py implementation"""
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
    """Standardized preprocessing matching api.py"""
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

def get_code_files():
    """Find all code files with proper language detection"""
    all_extensions = []
    for lang_config in LANGUAGE_CONFIG.values():
        all_extensions.extend(lang_config['extensions'])
    
    code_files = []
    for root, _, files in os.walk(REPO_DIR):
        for file in files:
            if any(file.endswith(ext) for ext in all_extensions):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        processed = preprocess_code(content)
                        if len(processed.split('\n')) >= PROCESSING_CONFIG['min_code_lines']:
                            code_files.append(full_path)
                except (UnicodeDecodeError, PermissionError, IOError) as e:
                    print(f"Skipping {full_path}: {str(e)}")
                    continue
    return code_files

def generate_embeddings(text, tokenizer, model, device):
    """Generate embeddings with error handling and proper batching"""
    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Attention-weighted mean pooling
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # Normalize to unit vectors
        embeddings = embeddings.cpu().numpy()
        return (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)).squeeze()
    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        return None

def create_index():
    """Main indexing function with progress tracking and validation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    code_files = get_code_files()
    if not code_files:
        print("No code files found. Check repository cloning and paths.")
        return
    
    print(f"Processing {len(code_files)} files...")
    
    embeddings = []
    metadata = []
    failed_files = 0
    
    for file in tqdm(code_files, desc="Generating embeddings"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine language from file extension
            language = 'python'  # default
            for lang, config in LANGUAGE_CONFIG.items():
                if any(file.endswith(ext) for ext in config['extensions']):
                    language = lang
                    break
            
            processed = preprocess_code(content, language)
            embedding = generate_embeddings(processed, tokenizer, model, device)
            
            if embedding is not None and not np.isnan(embedding).any():
                embeddings.append(embedding)
                metadata.append({
                    "path": file,
                    "language": language,
                    "original_length": len(content.split('\n')),
                    "processed_length": len(processed.split('\n'))
                })
            else:
                failed_files += 1
        except Exception as e:
            failed_files += 1
            continue
    
    if failed_files > 0:
        print(f"Failed to process {failed_files} files ({(failed_files/len(code_files))*100:.1f}%)")
    
    if not embeddings:
        raise ValueError("No valid embeddings generated - check your input files!")
    
    embeddings = np.array(embeddings).astype('float32')
    
    # Create and save index
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # Using Inner Product (cosine similarity) index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, os.path.join(INDEX_DIR, 'embeddings.faiss'))
    with open(os.path.join(INDEX_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nSuccessfully created index with {len(metadata)} embeddings")
    print(f"Index size: {embeddings.shape}")
    print(f"Saved to {INDEX_DIR}")

if __name__ == "__main__":
    clone_repositories()
    create_index()