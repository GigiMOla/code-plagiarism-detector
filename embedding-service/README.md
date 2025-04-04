### **2. embedding-service/README.md**
```markdown
# Embedding Service

Generates code embeddings using Microsoft's CodeBERT model.

## API Endpoints 📡
```http
POST /embed
Content-Type: application/json

{
  "text": "def add(a, b): return a + b",
  "language": "python"
}

Response:
{
  "embedding": [0.21, -0.45, ...]
}
Preprocessing Pipeline 🔄
Comment removal

Identifier normalization

Literal standardization

Import filtering

Type hint removal

Keyword preservation

Environment Variables 🌐
Variable	Default	Description
EMBEDDING_MODEL	microsoft/codebert-base	Transformer model name
DEVICE	auto	cuda/cpu
HF_HOME	/app/cache	HuggingFace cache
Performance Tuning ⚡
python

# For GPU acceleration
docker-compose build --build-arg DEVICE=cuda embedding-service
Development

bash

uvicorn app:app --reload --port 8001
Dependencies 📦
Transformers 4.31+

PyTorch 2.0+

FastAPI

---