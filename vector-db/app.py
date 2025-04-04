from fastapi import FastAPI, HTTPException
import numpy as np
import faiss
import pickle
import os
from pydantic import BaseModel

app = FastAPI()

class VectorAddRequest(BaseModel):
    embedding: list
    metadata: dict

class VectorSearchRequest(BaseModel):
    embedding: list
    k: int = 5

# Initialize FAISS index
index = None
metadata = []

@app.on_event("startup")
async def load_index():
    global index, metadata
    index_path = "/index/embeddings.faiss"
    metadata_path = "/index/metadata.pkl"
    
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        # Create new empty index
        index = faiss.IndexFlatIP(768)  # Default size for codebert
        os.makedirs("/index", exist_ok=True)

@app.post("/add")
async def add_vector(request: VectorAddRequest):
    try:
        embedding = np.array(request.embedding, dtype='float32').reshape(1, -1)
        index.add(embedding)
        metadata.append(request.metadata)
        
        # Persist changes
        faiss.write_index(index, "/index/embeddings.faiss")
        with open("/index/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/search")
async def search_vectors(request: VectorSearchRequest):
    try:
        embedding = np.array(request.embedding, dtype='float32').reshape(1, -1)
        distances, indices = index.search(embedding, request.k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                results.append({
                    "similarity": float(dist),
                    "metadata": metadata[idx]
                })
        
        return {"matches": results}
    except Exception as e:
        raise HTTPException(500, detail=str(e))