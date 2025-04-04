# Code Plagiarism Detection System 🔍

![Architecture Diagram](docs/architecture.png)

A distributed system for detecting code plagiarism across multiple programming languages using semantic similarity analysis.

## Features ✨
- Multi-language support (Python, Java, JavaScript)
- Code normalization and preprocessing
- CodeBERT-based embeddings
- FAISS vector similarity search
- Web-based interface with code comparison
- Automated evaluation framework
- Docker-based microservices architecture

## Services Overview 🛠️
| Service | Description | Port | Health Check |
|---------|-------------|------|--------------|
| Frontend | Web interface | 8000 | `/health` |
| Embedding | Code embedding generation | 8001 | `/health` |
| Vector DB | Vector storage & search | 8002 | `/health` |
| Indexing | Repository processing | - | - |
| Evaluation | Automated testing | - | - |

## Quick Start 🚀
```bash
# 1. Clone repository
git clone https://github.com/yourusername/code-plagiarism-detector.git
cd code-plagiarism-detector

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key

# 3. Start services
docker-compose up --build

# Access the web interface at http://localhost:8000
Configuration ⚙️
Edit .env for:

OPENAI_API_KEY: OpenAI API key for LLM analysis

SIMILARITY_THRESHOLD: Match threshold (0.6 recommended)

EMBEDDING_MODEL: CodeBERT model variant

REPO_FILE: List of repositories to index

Documentation 📚
Service	Documentation
Frontend	Web interface details
Embedding Service	Embedding generation
Vector DB	Vector storage details
Indexing Service	Repository processing
Evaluation	Testing framework
Development 🧑💻
bash

# Rebuild specific service
docker-compose build <service-name>

# View logs
docker-compose logs -f

# Run evaluation tests
docker-compose run evaluation