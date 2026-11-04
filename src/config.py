import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = os.getenv("KAGGLE_DATASET")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

DB_SERVER = os.getenv(
    "DB_SERVER", r"localhost\SQLEXPRESS"
)

DB_NAME = os.getenv("DB_NAME", "CarPriceDB")

DB_DRIVER = os.getenv(
    "DB_DRIVER", "ODBC Driver 18 for SQL Server"
)

DB_TRUSTED_CONNECTION = os.getenv(
    "DB_TRUSTED_CONNECTION", "yes"
)

DB_TRUST_SERVER_CERTIFICATE = os.getenv(
    "DB_TRUST_SERVER_CERTIFICATE", "no"
)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

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