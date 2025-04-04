import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment from .env file in project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Core settings - now with environment variable fallbacks
REPO_FILE = os.getenv("REPO_FILE", "config/repos.txt")
REPO_DIR = os.getenv("REPO_DIR", "repositories")
INDEX_DIR = os.getenv("INDEX_DIR", "index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "microsoft/codebert-base")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
TOP_K = int(os.getenv("TOP_K", "10"))
MIN_LLM_MATCH_LINES = int(os.getenv("MIN_LLM_MATCH_LINES", "20"))
MIN_CODE_LINES = int(os.getenv("MIN_CODE_LINES", "3"))

# Language configurations (unchanged from your original)
LANGUAGE_CONFIG = {
    'python': {
        'extensions': ['.py'],
        'comment_patterns': [r'#.*$', r'""".*?"""', r"'''.*?'''"],
        'type_hint_patterns': [r':\s*\w+\s*', r'->\s*\w+\s*'],
        'keywords': {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'finally', 'with', 'return', 'import', 'from', 'as', 'and', 'or', 'not',
            'is', 'in', 'pass', 'break', 'continue', 'raise', 'yield', 'async', 'await',
            'lambda', 'nonlocal', 'global', 'True', 'False', 'None'
        }
    },
    'java': {
        'extensions': ['.java'],
        'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
        'type_hint_patterns': [],
        'keywords': {
            'public', 'private', 'protected', 'class', 'interface', 'abstract',
            'static', 'final', 'void', 'int', 'long', 'float', 'double', 'char',
            'boolean', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
            'try', 'catch', 'finally', 'throw', 'throws', 'return', 'new', 'this',
            'super', 'extends', 'implements', 'package', 'import', 'true', 'false', 'null'
        }
    },
    'javascript': {
        'extensions': ['.js', '.jsx'],
        'comment_patterns': [r'//.*$', r'/\*.*?\*/'],
        'type_hint_patterns': [],
        'keywords': {
            'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'do',
            'switch', 'case', 'default', 'try', 'catch', 'finally', 'throw', 'return',
            'new', 'this', 'class', 'extends', 'async', 'await', 'export', 'import',
            'true', 'false', 'null', 'undefined', 'typeof', 'instanceof', 'in', 'of',
            'delete'
        }
    }
}

# Processing config with env fallbacks
PROCESSING_CONFIG = {
    'max_code_lines': int(os.getenv("MAX_CODE_LINES", "100")),
    'min_code_lines': int(os.getenv("MIN_CODE_LINES", "3")),
    'preserve_imports': os.getenv("PRESERVE_IMPORTS", "false").lower() == "true"
}

# New service configuration
SERVICE_CONFIG = {
    'embedding_service_url': os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001"),
    'vector_db_url': os.getenv("VECTOR_DB_URL", "http://vector-db:8002"),
    'frontend_url': os.getenv("FRONTEND_URL", "http://frontend:8000")
}