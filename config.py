"""Configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
PERSIST_DIR = os.getenv("PERSIST_DIR", str(BASE_DIR / "chroma_db"))
REPOS_DIR = os.getenv("REPOS_DIR", str(BASE_DIR / "repos"))
GRAPH_DIR = os.getenv("GRAPH_DIR", str(BASE_DIR / "graph_db"))
BM25_DIR = str(BASE_DIR/"bm25_db")

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


for d in [PERSIST_DIR, REPOS_DIR, GRAPH_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CONTEXT_TOKENS = 12000
CHUNK_SIZE = 1500

DIAGRAM_KEYWORDS = [
    r'\bflow\b', r'\bdiagram\b', r'\bvisuali[sz]e\b', r'\barchitecture\b',
    r'\bstructure\b', r'\bhow\s+does\b', r'\bsequence\b', r'\bprocess\b',
    r'\bpipeline\b', r'\bworkflow\b', r'\bcall\s*graph\b', r'\bdependenc'
]

IGNORE_PATTERNS = [
    r'\.git/', r'node_modules/', r'__pycache__/', r'venv/', r'\.venv/',
    r'dist/', r'build/', r'target/', r'\.idea/', r'\.vscode/',
    r'\.min\.js$', r'\.min\.css$', r'package-lock\.json$', r'yarn\.lock$',
    r'go\.sum$', r'\.class$', r'\.pyc$', r'\.o$', r'\.so$',
    r'\.png$', r'\.jpg$', r'\.gif$', r'\.pdf$', r'\.zip$', r'\.tar$',
    r'vendor/', r'third_party/', r'\.pb\.go$'
]

# Language detection
LANG_MAP = {
    '.py': 'python',
    '.js': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
    '.jsx': 'jsx',
    '.ts': 'typescript', '.mts': 'typescript',
    '.tsx': 'tsx',
    '.java': 'java',
    '.go': 'go',
    '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml',
    '.md': 'markdown', '.rst': 'markdown', '.txt': 'text',
    '.html': 'html', '.css': 'css', '.scss': 'scss',
    '.sql': 'sql', '.sh': 'bash', '.bash': 'bash',
    '.rb': 'ruby', '.rs': 'rust', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
    '.cs': 'csharp', '.php': 'php', '.swift': 'swift', '.kt': 'kotlin',
}

# Languages with full AST support
AST_LANGUAGES = {'python', 'javascript', 'jsx', 'typescript', 'tsx', 'java', 'go'}

MAX_FILE_SIZE = 1024 * 1024  # 1MB
