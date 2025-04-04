### **3. frontend/README.md**
```markdown
# Frontend Service

Web interface for plagiarism detection system.

## Routes 🗺️
| Path | Method | Description |
|------|--------|-------------|
| `/` | GET | Submission form |
| `/check` | POST | Process code |
| `/health` | GET | Service status |

## Templates 🎨
| Template | Purpose |
|----------|---------|
| index.html | Code submission form |
| results.html | Match visualization |
| error.html | Error handling |

## Detection Methods 🔍
1. **RAG (Vector Search)**
   - Pure semantic similarity
   - Fastest method
2. **LLM Analysis**
   - GPT-based deep analysis
   - Most accurate
3. **Combined**
   - Hybrid approach (recommended)

## Environment Variables 🌐
| Variable | Description |
|----------|-------------|
| EMBEDDING_SERVICE_URL | http://embedding-service:8001 |
| VECTOR_DB_URL | http://vector-db:8002 |
| OPENAI_API_KEY | OpenAI API key |

## Development
```bash
uvicorn app:app --reload --port 8000
UI Features 🖥️
Code comparison view

Similarity score visualization

Multiple detection methods

Language selection