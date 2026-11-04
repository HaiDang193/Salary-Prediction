import pandas as pd
import pyodbc

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from src.config import (
    DB_SERVER,
    DB_NAME,
    DB_DRIVER,
    DB_TRUSTED_CONNECTION,
    DB_TRUST_SERVER_CERTIFICATE,
    DB_USER,
    DB_PASSWORD,
    EXPECTED_COLUMNS,
)


def get_engine():
    connection_parts = [
        f"DRIVER={{{DB_DRIVER}}}",
        f"SERVER={DB_SERVER}",
        f"DATABASE={DB_NAME}",
        "Encrypt=yes",
        f"TrustServerCertificate={DB_TRUST_SERVER_CERTIFICATE}",
    ]

    if DB_TRUSTED_CONNECTION.lower() == "yes":
        connection_parts.append("Trusted_Connection=yes")
    else:
        if not DB_USER or not DB_PASSWORD:
            raise ValueError(
                "DB_USER and DB_PASSWORD are required"
            )

        connection_parts.extend([
            f"UID={DB_USER}",
            f"PWD={DB_PASSWORD}",
        ])

    connection_string = ";".join(connection_parts) + ";"

    url = URL.create(
        "mssql+pyodbc",
        query={"odbc_connect": connection_string},
    )

    return create_engine(
        url,
        pool_pre_ping=True,
    )


def test_connection():
    engine = get_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT DB_NAME()")
            ).scalar_one()

            print(f"[CONNECTED] Database: {result}")
    finally:
        engine.dispose()


def import_cars(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Cannot import empty dataset")

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "DataFrame schema does not match cars_raw"
        )

    engine = get_engine()

    try:
        # One transaction: delete + insert
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM dbo.cars_raw")
            )

            df.to_sql(
                name="cars_raw",
                con=conn,
                schema="dbo",
                if_exists="append",
                index=False,
                chunksize=500,
            )

            count = conn.execute(
                text("SELECT COUNT(*) FROM dbo.cars_raw")
            ).scalar_one()

            if count != len(df):
                raise RuntimeError(
                    "Row count mismatch after import"
                )

        print(f"[SUCCESS] Imported {count:,} rows")

    finally:
        engine.dispose()


def load_cars_from_sql() -> pd.DataFrame:
    engine = get_engine()

    try:
        query = text("""
            SELECT
                name,
                year,
                selling_price,
                km_driven,
                fuel,
                seller_type,
                transmission,
                owner
            FROM dbo.cars_raw
        """)

        with engine.connect() as conn:
            return pd.read_sql(query, conn)

    finally:
        engine.dispose()