import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.config import (
    RAW_DIR,
    KAGGLE_DATASET,
    KAGGLE_API_TOKEN,
)


def download_dataset(filename: str, force=False) -> Path:
    # 1. Check configuration
    if not KAGGLE_DATASET or "/" not in KAGGLE_DATASET:
        raise ValueError(
            "Please configure KAGGLE_DATASET in .env"
        )

    if not KAGGLE_API_TOKEN:
        raise ValueError(
            "Please configure KAGGLE_API_TOKEN in .env"
        )

    if not filename.lower().endswith(".csv"):
        raise ValueError("Filename must be a CSV file")

    # 2. Output file
    target_dir = RAW_DIR
    output_file = target_dir / "car_dataset.csv"

    # 3. Skip if file exists
    if output_file.exists() and not force:
        print("[SKIP] Dataset already downloaded")
        return output_file

    # 4. Kaggle authentication
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

    # 5. Download selected CSV
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)

        command = [
            sys.executable,
            "-m", "kaggle",
            "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-f", filename,
            "-p", str(temp_dir),
            "--unzip",
        ]

        print(f"[DOWNLOAD] {KAGGLE_DATASET}")
        print(f"[FILE] {filename}")

        subprocess.run(
            command,
            check=True,
            env=env,
        )

        # 6. Find downloaded CSV
        csv_files = list(temp_dir.rglob("*.csv"))

        if len(csv_files) != 1:
            raise FileNotFoundError(
                f"Expected 1 CSV, found {len(csv_files)}"
            )

        # 7. Save selected CSV
        target_dir.mkdir(parents=True, exist_ok=True)

        temp_output = target_dir / "car_dataset.csv.tmp"

        try:
            shutil.copy2(csv_files[0], temp_output)
            os.replace(temp_output, output_file)
        finally:
            if temp_output.exists():
                temp_output.unlink()

    print("[SUCCESS] Kaggle dataset downloaded")
    print(f"[SAVED] {output_file}")

    return output_file