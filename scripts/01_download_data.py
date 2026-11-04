from src.kaggle_client import download_dataset

# Configuration
CSV_FILENAME = "CAR DETAILS FROM CAR DEKHO.csv"
FORCE_DOWNLOAD = False


def main():
    download_dataset(
        filename=CSV_FILENAME,
        force=FORCE_DOWNLOAD
    )


if __name__ == "__main__":
    main()
