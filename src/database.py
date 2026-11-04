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


def get_engine():
    url = URL.create(
        drivername="postgresql+psycopg",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        query={"sslmode": "require"},
    )

    return create_engine(url, pool_pre_ping=True)


def test_connection():
    engine = get_engine()

    try:
        with engine.connect() as conn:
            db_name = conn.execute(
                text("SELECT current_database()")
            ).scalar_one()

            print(f"[CONNECTED] Supabase: {db_name}")
    finally:
        engine.dispose()


def import_cars(df: pd.DataFrame):
    engine = get_engine()

    try:
        with engine.begin() as conn:
            # Import only when raw table is empty
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
    finally:
        engine.dispose()


def load_cars_from_sql():
    engine = get_engine()

    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT name, year, selling_price,
                           km_driven, fuel, seller_type,
                           transmission, owner
                    FROM public.cars_raw
                """),
                conn,
            )
    finally:
        engine.dispose()