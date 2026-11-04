from src.config import RAW_DIR
from src.data_loader import (
    find_dataset_csv,
    load_csv,
)
from src.database import (
    test_connection,
    import_cars,
)


def main():
    print("=" * 50)
    print("IMPORT CAR DATA TO SQL SERVER")
    print("=" * 50)

    # Step 1: Find CSV
    csv_path = find_dataset_csv(
        RAW_DIR / "kaggle"
    )

    print(f"[FILE] {csv_path}")

    # Step 2: Read and validate
    df = load_csv(csv_path)

    print(f"[ROWS] {len(df):,}")
    print(f"[COLUMNS] {len(df.columns)}")

    # Step 3: Test SQL connection
    test_connection()

    # Step 4: Import data
    import_cars(df)

    print("[DONE] Import completed")


if __name__ == "__main__":
    main()