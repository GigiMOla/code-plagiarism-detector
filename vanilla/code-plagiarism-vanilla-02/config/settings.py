import os
from dotenv import load_dotenv

load_dotenv()

# Core settings
REPO_FILE = "config/repos.txt"
REPO_DIR = "repositories"
INDEX_DIR = "index"
EMBEDDING_MODEL = "microsoft/codebert-base"
GPT_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIMILARITY_THRESHOLD = 0.6
TOP_K = 10
MIN_LLM_MATCH_LINES = 20
MIN_CODE_LINES = 3

# Language configurations
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

PROCESSING_CONFIG = {
    'max_code_lines': 100,
    'min_code_lines': 2,
    'preserve_imports': False
}