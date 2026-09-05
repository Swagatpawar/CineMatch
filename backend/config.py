import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
FRONTEND_URL = os.getenv("FRONTEND_URL", "").strip()
if FRONTEND_URL and FRONTEND_URL not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(FRONTEND_URL)

MODEL_PATH = BASE_DIR / "models" / "cinematch_svd_final.pkl"
MOVIES_PATH = BASE_DIR / "data" / "movies.csv"
RATINGS_PATH = BASE_DIR / "data" / "ratings.csv"

POPULAR_MIN_RATINGS = 50
TOP_RATED_MIN_RATINGS = 100
DEFAULT_RECOMMENDATION_LIMIT = 5
MAX_RECOMMENDATION_LIMIT = 50


def get_api_base_url() -> str:
    return os.getenv("VITE_API_URL", "http://localhost:8000")
