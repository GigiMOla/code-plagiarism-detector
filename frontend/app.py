from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import httpx
import os
from config.settings import LANGUAGE_CONFIG

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://vector-db:8002")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "languages": list(LANGUAGE_CONFIG.keys())
    })

@app.post("/check")
async def check_plagiarism(request: Request, code: str = Form(...), language: str = Form(...)):
    try:
        # Get embedding
        async with httpx.AsyncClient() as client:
            # Get embedding first
            embed_resp = await client.post(
                f"{EMBEDDING_SERVICE_URL}/embed",
                json={"text": code, "language": language},
                timeout=30
            )
            embedding = embed_resp.json()["embedding"]

            # Search vector DB
            search_resp = await client.post(
                f"{VECTOR_DB_URL}/search",
                json={"embedding": embedding, "k": 5},
                timeout=30
            )
            matches = search_resp.json().get("matches", [])

            # Get file contents for matches
            results = []
            for match in matches:
                with open(match["metadata"]["path"], 'r', encoding='utf-8') as f:
                    results.append({
                        "file": match["metadata"]["path"],
                        "similarity": match["similarity"],
                        "code": f.read()
                    })

            # Determine if plagiarized
            plagiarized = any(m['similarity'] >= SIMILARITY_THRESHOLD for m in results)
            top_similarity = max([m['similarity'] for m in results]) if results else 0.0

        return templates.TemplateResponse("results.html", {
            "request": request,
            "input_code": code,
            "language": language,
            "results": results,
            "plagiarized": plagiarized,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "top_similarity": top_similarity
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })