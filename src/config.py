import os
from pathlib import Path

from dotenv import load_dotenv


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. KAGGLE CONFIGURATION
# ==========================================

KAGGLE_DATASET = os.getenv("KAGGLE_DATASET")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")


# ==========================================
# 3. SUPABASE POSTGRESQL CONFIGURATION
# ==========================================

DB_HOST = os.getenv("DB_HOST")

DB_PORT = int(os.getenv("DB_PORT", "5432"))

DB_NAME = os.getenv("DB_NAME", "postgres")

DB_USER = os.getenv("DB_USER")

DB_PASSWORD = os.getenv("DB_PASSWORD")


# ==========================================
# 4. DATASET SCHEMA
# ==========================================

EXPECTED_COLUMNS = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]