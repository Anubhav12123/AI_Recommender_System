import os

def get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

LEXICAL_WEIGHT = get_float("LEXICAL_WEIGHT", 0.5)
VECTOR_WEIGHT  = get_float("VECTOR_WEIGHT", 0.3)
CF_WEIGHT      = get_float("CF_WEIGHT", 0.2)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

ARTIFACT_DIR = "data/artifacts"
ITEMS_CSV = "data/raw/items.csv"
RATINGS_CSV = "data/raw/ratings.csv"
