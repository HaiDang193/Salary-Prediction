from src.config import RAW_DIR
from src.data_loader import load_csv
from src.database import test_connection, import_cars


def main():
    csv_path = RAW_DIR / "car_dataset.csv"

    df = load_csv(csv_path)

    test_connection()

    import_cars(df)


if __name__ == "__main__":
    main()