import requests
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# Đọc file .env
load_dotenv(BASE_DIR / ".env")

app_id = os.environ["ADZUNA_APP_ID"]
app_key = os.environ["ADZUNA_APP_KEY"]
# 2. API Search việc làm tại Mỹ (us)
url_goc = "https://api.adzuna.com/v1/api/jobs/us/search/"


danh_sach_tu_khoa = [
    "Software Engineer",
    "Data Scientist",
    "DevOps",
    "Machine Learning",
    "Cloud Architect",
    "Cybersecurity"
]
danh_sach_dia_diem = [
    "California",
    "New York",
    "Texas",
    "Washington",
    "Florida"
]
# 3. Áp dụng các tham số lọc phức tạp
params = {
    "app_id": app_id,
    "app_key": app_key,
    "results_per_page": 50,
    "full_time": 1,
    "permanent": 1,
    "content-type": "application/json"
}
# Tạo một danh sách rỗng để chứa tất cả dữ liệu
all_jobs = []

# 2. Vòng lặp quét qua từng từ khóa và địa điểm
for tu_khoa in danh_sach_tu_khoa:
    for dia_diem in danh_sach_dia_diem:
        for trang in range(1, 6):
            print(f"Đang tìm: {tu_khoa} tại {dia_diem} - TRANG {trang}...")

            # Ghép URL với số trang (Ví dụ: .../search/1, .../search/2)
            url_truy_van = url_goc + str(trang)

            params["what"] = tu_khoa
            params["where"] = dia_diem

            response = requests.get(url_truy_van, params=params)

            if response.status_code == 200:
                data = response.json()
                jobs_list = data.get("results", [])

                # Nếu trang này không có dữ liệu (hết tin), thì DỪNG vòng lặp trang này lại
                if len(jobs_list) == 0:
                    print(f" Đã hết tin tuyển dụng ở trang {trang}. Chuyển sang bang khác!")
                    break

                all_jobs.extend(jobs_list)
            else:
                print(f"Lỗi {response.status_code}")
                break

            time.sleep(4)

ROOT = Path.cwd()
DATA_DIR_SAVE = ROOT / ".." / "data" / "raw"
os.makedirs(DATA_DIR_SAVE, exist_ok=True)
ten_file = os.path.join(DATA_DIR_SAVE, "du_lieu_adzuna_tho.json")
with open(ten_file, "w", encoding="utf-8") as f:
    json.dump(all_jobs, f, ensure_ascii=False, indent=4)
print(f"Đã lưu {len(all_jobs)} bản ghi vào: {ten_file}")