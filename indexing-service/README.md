5. indexing-service/README.md
markdown
Copy
# Indexing Service

Processes repositories into the vector database.

## Workflow 🔄
1. Clone repositories from `config/repos.txt`
2. Discover code files (Python/Java/JS)
3. Generate embeddings
4. Store in vector DB

## File Requirements 📄
- Minimum 3 lines of code
- Valid file extensions:
  - .py (Python)
  - .java (Java)
  - .js/.jsx (JavaScript)

## Environment Variables 🌐
| Variable | Default | Description |
|----------|---------|-------------|
| REPO_DIR | repositories | Clone directory |
| GIT_TIMEOUT | 300 | Clone timeout (seconds) |

## Manual Execution
```bash
python index.py
Monitoring 🕵️
Progress bar with tqdm

Error logging for failed files

Automatic retry for service dependencies