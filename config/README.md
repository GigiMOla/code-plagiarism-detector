7. config/README.md
markdown
Copy
# Configuration Center

Central configuration management for all services.

## Files 📄
| File | Purpose |
|------|---------|
| repos.txt | Repository URLs to index |
| settings.py | Core system configuration |
| .env | Environment variables |

## Key Settings ⚙️
```python
# Detection parameters
SIMILARITY_THRESHOLD = 0.6  # 0-1 scale
MIN_CODE_LINES = 3
MAX_CODE_LINES = 100

# Language configurations
LANGUAGE_CONFIG = {
    'python': {
        'extensions': ['.py'],
        'keywords': ['def', 'class', ...]
    }
    # ... other languages
}
Environment Management 🌐
Loaded via python-dotenv

Fallback to defaults in settings.py

Hierarchical override: ENV > .env > settings.py