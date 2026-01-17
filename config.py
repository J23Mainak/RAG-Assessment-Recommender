import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

PRODUCTS_JSON = DATA_DIR / "shl_assessments_complete.json"
PRODUCTS_CSV = DATA_DIR / "products.csv"
TRAIN_SET = DATA_DIR / "train_set.csv"
TEST_SET = DATA_DIR / "test_set.csv"

DATA_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set!")

GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_EMBED_MODEL = "models/text-embedding-004"  # Gemini embeddings
GEMINI_RERANK_MODEL = "gemini-2.5-pro"  # For re-ranking

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT = 30

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "shl_assessments_gemini"
QDRANT_USE_MEMORY = True  

# Embedding dimensions for Gemini text-embedding-004
EMBEDDING_DIM = 768

# Search parameters
QDRANT_SEARCH_LIMIT = 50
RERANK_TOP_K = 30
FINAL_K_MIN = 5
FINAL_K_MAX = 10

WEIGHT_SEMANTIC = 0.40
WEIGHT_LLM_RERANK = 0.50  # Higher because it's more accurate
WEIGHT_METADATA = 0.10

MIN_TECHNICAL_TESTS = 3
MIN_BEHAVIORAL_TESTS = 2

TEST_TYPE_MAPPING = {
    'K': 'Knowledge & Technical Skills',
    'P': 'Personality & Behavioral',
    'B': 'Situational Judgement',
    'S': 'Simulations',
    'A': 'Cognitive Ability',
    'C': 'Competencies',
    'D': 'Development & 360 Feedback',
    'E': 'Assessment Exercises'
}

API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("PORT", 8000))

ENABLE_CACHE = True
CACHE_SIZE = 512

print(f"Config loaded: GEMINI_MODEL={GEMINI_MODEL}")
print(f"Embeddings: {GEMINI_EMBED_MODEL}")
print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")