
from src.config import RAW_DIR
from src.data_loader import load_csv
from src.data_validation import validate_cars


def main():
    df = load_csv(RAW_DIR / "car_dataset.csv")
    results = validate_cars(df)

    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
