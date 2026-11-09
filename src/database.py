
import time
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from src.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    EXPECTED_COLUMNS,
)


@lru_cache(maxsize=1)
def get_engine():
    """
    Create one reusable SQLAlchemy Engine.
    """
    url = URL.create(
        drivername="postgresql+psycopg",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        query={"sslmode": "require"},
    )

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=2,
        pool_timeout=10,
        connect_args={
            "connect_timeout": 10,
        },
    )


def test_connection():
    engine = get_engine()

    start = time.perf_counter()

    with engine.connect() as conn:
        db_name = conn.execute(
            text("SELECT current_database()")
        ).scalar_one()

    elapsed = time.perf_counter() - start

    print(f"[CONNECTED] Supabase: {db_name}")
    print(f"[TIME] {elapsed:.2f}s")


def import_cars(df: pd.DataFrame):
    engine = get_engine()

    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT COUNT(*) FROM public.cars_raw")
        ).scalar_one()

        if existing > 0:
            print(f"[SKIP] Table already has {existing} rows")
            return

        df.to_sql(
            name="cars_raw",
            con=conn,
            schema="public",
            if_exists="append",
            index=False,
            chunksize=500,
        )

        print(f"[IMPORTED] {len(df):,} rows")


def load_cars_from_sql():
    engine = get_engine()

    start = time.perf_counter()

    print("[1/3] Connecting to Supabase...", flush=True)

    with engine.connect() as conn:

        connected_at = time.perf_counter()

        print(
            f"[2/3] Connected in "
            f"{connected_at - start:.2f}s",
            flush=True,
        )

        print("[3/3] Loading data...", flush=True)

        df = pd.read_sql(
            text("""
                SELECT
                    name,
                    year,
                    selling_price,
                    km_driven,
                    fuel,
                    seller_type,
                    transmission,
                    owner
                FROM public.cars_raw
            """),
            conn,
        )

    finished_at = time.perf_counter()

    print(f"[SUCCESS] Loaded {len(df):,} rows")
    print(
        f"[QUERY] {finished_at - connected_at:.2f}s"
    )
    print(f"[TOTAL] {finished_at - start:.2f}s")

    return df
