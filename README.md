
# Car Price Prediction — Dự đoán giá ô tô cũ

Dự án Machine Learning dự đoán **giá bán ô tô đã qua sử dụng** dựa trên các thông tin về xe. Dự án sử dụng dữ liệu **CarDekho**, thử nghiệm các mô hình hồi quy và triển khai mô hình **XGBoost** đã huấn luyện trên giao diện **Streamlit**.

> **Đơn vị dự đoán:** INR (rupee Ấn Độ). Đây là giá ước lượng bằng mô hình, không phải báo giá thị trường và không phải VND.

## 1. Mục tiêu dự án

- Khám phá dữ liệu và xác định các yếu tố liên quan đến giá bán xe cũ.
- Làm sạch dữ liệu, xây dựng đặc trưng và so sánh các mô hình hồi quy.
- Tối ưu XGBoost bằng `RandomizedSearchCV` và đánh giá trên tập test.
- Lưu model để dự đoán mà không cần huấn luyện lại mỗi lần sử dụng.
- Xây dựng ứng dụng Streamlit cho phép nhập thông tin xe và nhận kết quả dự đoán.

## 2. Bộ dữ liệu

**Nguồn:** [Vehicle Dataset from CarDekho — Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

- Tệp gốc trên Kaggle: `CAR DETAILS FROM CAR DEKHO.csv`
- Tệp trong dự án: `data/raw/car_dataset.csv`
- Kích thước ban đầu: **4.340 dòng, 8 cột**
- Dữ liệu thiếu: Không phát hiện giá trị null.
- Bản ghi trùng hoàn toàn: **763**
- Sau khi loại trùng: **3.577 dòng**

### Các thuộc tính

| Cột | Mô tả |
|---|---|
| `name` | Hãng xe, mẫu xe và phiên bản |
| `year` | Năm sản xuất |
| `selling_price` | Giá bán thực tế (INR) — biến mục tiêu |
| `km_driven` | Số km xe đã đi |
| `fuel` | Loại nhiên liệu |
| `seller_type` | Loại người bán |
| `transmission` | Loại hộp số |
| `owner` | Thông tin số đời chủ sở hữu |

## 3. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| Pandas, NumPy | Xử lý và biến đổi dữ liệu |
| Matplotlib, Seaborn | Trực quan hóa và EDA |
| Scikit-learn | Chia dữ liệu, Cross-Validation, tuning và đánh giá |
| XGBoost | Mô hình dự đoán giá |
| Joblib | Lưu và tải model |
| Streamlit | Xây dựng giao diện dự đoán |
| Supabase PostgreSQL | Lưu trữ dữ liệu dùng chung |
| SQLAlchemy, Psycopg | Kết nối PostgreSQL |
| Kaggle API | Tải bộ dữ liệu |
| Jupyter Notebook | Phân tích và thử nghiệm mô hình |
| Git, GitHub | Quản lý phiên bản |

## 4. Cấu trúc dự án

```text
Salary Prediction/
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── car_dataset.csv
│   └── processed/
│       └── car_processed.csv
│
├── database/
│   └── 01_create_tables.sql
│
├── models/
│   └── car_price_xgboost.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_model.ipynb
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_import_data.py
│   ├── 03_validate_data.py
│   ├── 04_train_model.py
│   └── 05_test_prediction.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── kaggle_client.py
│   ├── database.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### Chức năng các thư mục

- `data/`: Chứa dữ liệu gốc và dữ liệu đã xử lý.
- `database/`: Chứa SQL tạo bảng trên Supabase.
- `notebooks/`: Chứa các bước phân tích, tiền xử lý và thử nghiệm.
- `src/`: Chứa logic xử lý dữ liệu, huấn luyện và dự đoán.
- `scripts/`: Chứa các lệnh thực thi độc lập.
- `models/`: Chứa model đã huấn luyện.
- `app/`: Chứa giao diện Streamlit.

## 5. Quy trình Machine Learning

### 5.1. Data Understanding

Thực hiện trong `01_data_understanding.ipynb`.

Các công việc:

- Tìm hiểu nguồn dữ liệu và ý nghĩa các thuộc tính.
- Kiểm tra kích thước và kiểu dữ liệu.
- Phát hiện giá trị thiếu.
- Kiểm tra bản ghi trùng lặp.
- Thống kê mô tả các biến số và biến phân loại.

**Kết quả:** Dataset có 4.340 bản ghi, 8 thuộc tính, không có giá trị null và có 763 bản ghi trùng hoàn toàn.

### 5.2. Exploratory Data Analysis (EDA)

Thực hiện trong `02_eda.ipynb`.

Các nội dung phân tích:

**Univariate Analysis**

Phân tích phân phối của `selling_price`, `km_driven` và các đặc trưng khác.

**Outlier Analysis**

Sử dụng Boxplot để quan sát các giá trị ngoại lai trong giá bán và số km.

**Bivariate Analysis**

Phân tích sự khác biệt về giá bán theo nhiên liệu, hộp số, người bán và chủ sở hữu.

**Correlation Analysis**

Sử dụng Pearson Correlation và Heatmap để đánh giá mối tương quan tuyến tính giữa các biến số.

### 5.3. Data Preprocessing

Thực hiện trong `03_preprocessing.ipynb`.

**Data Cleaning**

Loại bỏ các bản ghi trùng lặp hoàn toàn.

```python
df = df.drop_duplicates().reset_index(drop=True)
```

**Feature Engineering**

Tạo đặc trưng tuổi xe:

```python
REFERENCE_YEAR = 2026

df["vehicle_age"] = REFERENCE_YEAR - df["year"]
```

Trích xuất hãng xe từ tên xe để tạo đặc trưng `brand`.

**Target Transformation**

Giảm độ lệch phải của biến mục tiêu bằng Log Transformation:

```python
df["log_selling_price"] = np.log1p(
    df["selling_price"]
)
```

**Categorical Encoding**

- `owner`: Ordinal Encoding.
- `brand`: One-Hot Encoding.
- `fuel`: One-Hot Encoding.
- `seller_type`: One-Hot Encoding.
- `transmission`: One-Hot Encoding.

Sử dụng `drop_first=True` khi tạo các biến dummy.

**Kết quả preprocessing:**

- Số dòng: 3.577
- Số features: 38
- Số target: 1
- File đầu ra: `data/processed/car_processed.csv`

### 5.4. Model Training

Thực hiện trong `04_model.ipynb`.

Dữ liệu được chia theo tỷ lệ:

- Training set: 80% (2.861 dòng)
- Testing set: 20% (716 dòng)
- Random state: 42

Hai mô hình được thử nghiệm:

**Linear Regression**

Sử dụng làm mô hình baseline để so sánh hiệu quả dự đoán.

**XGBoost Regressor**

Sử dụng thuật toán Gradient Boosting để học các quan hệ phi tuyến giữa đặc trưng và giá xe.

Quá trình so sánh sử dụng **5-Fold Cross-Validation**.

### 5.5. Hyperparameter Tuning

Sử dụng `RandomizedSearchCV` để tối ưu XGBoost.

Các tham số được tìm kiếm:

| Hyperparameter | Giá trị thử nghiệm |
|---|---|
| `n_estimators` | 100, 200, 300, 400, 500 |
| `learning_rate` | 0.01, 0.05, 0.1 |
| `max_depth` | 3, 5, 7 |
| `subsample` | 0.8, 1.0 |
| `colsample_bytree` | 0.8, 1.0 |

Thiết lập:

- `n_iter = 180`
- `cv = 5`
- `random_state = 42`
- `refit = "r2"`

**Best Parameters tìm được:**

```python
{
    "subsample": 0.8,
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8
}
```

### 5.6. Model Evaluation

Kết quả được ghi nhận trong notebook `04_model.ipynb`.

**So sánh Cross-Validation**

| Model | CV R² | CV MAE | CV RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.7760 | 0.3014 | 0.3845 |
| XGBoost Baseline | 0.7609 | 0.3077 | 0.3971 |
| XGBoost Tuned | **0.7881** | **0.2902** | **0.3738** |

Các metric trong bảng trên được tính trên thang log-price.

**Kết quả trên tập Test**

| Metric | Giá trị |
|---|---:|
| R² (log-price) | 0.7982 |
| MAE (log-price) | 0.2893 |
| RMSE (log-price) | 0.3720 |
| MAE (INR) | 140,910.52 |
| RMSE (INR) | 334,868.79 |

Để đánh giá trên thang giá gốc, kết quả dự đoán được chuyển đổi bằng:

```python
y_pred_real = np.expm1(y_pred)
```

**Mô hình được lựa chọn:** XGBoost Regressor sau Hyperparameter Tuning.

Các kết quả trên phản ánh lần thử nghiệm đã lưu trong notebook, chưa phải đánh giá lại trên dữ liệu mới.

## 6. Lưu trữ model

Sau khi hoàn tất huấn luyện, model được lưu bằng Joblib:

```python
import joblib
from pathlib import Path

MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    best_xgb,
    MODEL_DIR / "car_price_xgboost.joblib"
)
```

Model được lưu tại:

```text
models/car_price_xgboost.joblib
```

Ứng dụng Streamlit sử dụng lại model này mà không phải huấn luyện lại.

Tên và thứ tự các features được lấy từ:

```python
model.feature_names_in_
```

Do đó, **không cần file `model_features.json`** nếu model có thuộc tính này.

## 7. Cài đặt môi trường

### 7.1. Clone repository

```powershell
git clone <repository-url>
```

Di chuyển vào thư mục dự án.

### 7.2. Tạo môi trường ảo

Nếu chưa có `.venv`:

```powershell
python -m venv .venv
```

Kích hoạt trên PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu đã có `.venv`, chỉ cần kích hoạt lại, không cần tạo môi trường mới.

### 7.3. Cài đặt thư viện

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra thư viện:

```powershell
python -m pip check
```

## 8. Chạy ứng dụng Streamlit

Đảm bảo file model đã tồn tại tại đường dẫn:

```text
models/car_price_xgboost.joblib
```

Từ thư mục gốc của dự án, chạy:

```powershell
python -m streamlit run app/streamlit_app.py
```

Streamlit sẽ khởi chạy ứng dụng trên trình duyệt.

### Thông tin đầu vào

Người dùng nhập:

- Tên xe
- Năm sản xuất
- Số km đã đi
- Loại nhiên liệu
- Loại người bán
- Hộp số
- Thông tin chủ sở hữu

Ví dụ:

```text
Name: Maruti Swift Dzire VDI
Year: 2018
KM Driven: 50000
Fuel: Diesel
Seller Type: Individual
Transmission: Manual
Owner: First Owner
```

Nhấn **Predict Price** để nhận kết quả.

### Prediction Pipeline

```text
User Input
    |
    v
Streamlit Form
    |
    v
src/preprocessing.py
    |
    v
Feature Engineering
    |
    v
38 Model Features
    |
    v
src/predict.py
    |
    v
XGBoost Model
    |
    v
Log Price Prediction
    |
    v
np.expm1()
    |
    v
Predicted Price (INR)
```

Các bước tiền xử lý khi dự đoán phải giống với quy trình đã sử dụng lúc huấn luyện model.

Ứng dụng dự đoán không yêu cầu kết nối Supabase vì model được tải trực tiếp từ file `.joblib`.

## 9. Chạy các Scripts

Tất cả lệnh được thực hiện từ thư mục gốc dự án.

### 9.1. Download Dataset

```powershell
python -m scripts.01_download_data
```

Tải CSV từ Kaggle vào:

```text
data/raw/car_dataset.csv
```

### 9.2. Import Data to Supabase

```powershell
python -m scripts.02_import_data
```

Nạp dữ liệu vào bảng `public.cars_raw`.

Cần tạo bảng bằng file `database/01_create_tables.sql` trước khi import.

### 9.3. Validate Dataset

```powershell
python -m scripts.03_validate_data
```

Kiểm tra cấu trúc dữ liệu, giá trị thiếu, trùng lặp và giá bán không hợp lệ.

### 9.4. Train Model

```powershell
python -m scripts.04_train_model
```

Huấn luyện và tuning lại XGBoost.

**Lưu ý:** Script này có thể ghi đè model hiện tại. Không cần chạy nếu chỉ muốn sử dụng model đã train.

### 9.5. Test Prediction

```powershell
python -m scripts.05_test_prediction
```

Kiểm tra quá trình tải model, preprocessing và dự đoán.

## 10. Cấu hình Supabase và Kaggle

Tạo file `.env` dựa trên `.env.example`.

```dotenv
# Kaggle
KAGGLE_DATASET=nehalbirla/vehicle-dataset-from-cardekho
KAGGLE_API_TOKEN=your_kaggle_api_token

# Supabase PostgreSQL
DB_HOST=your-session-pooler-host
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres.your_project_ref
DB_PASSWORD=your_database_password
```

Thông tin kết nối được sử dụng bởi:

- `src/config.py`
- `src/kaggle_client.py`
- `src/database.py`

**Bảo mật:** Không commit `.env`, API token hoặc database password lên GitHub.

Supabase được sử dụng để lưu trữ và truy xuất dữ liệu. Ứng dụng Streamlit hiện tại chưa sử dụng database để lưu lịch sử dự đoán.

## 11. Troubleshooting

### Lỗi: No module named streamlit

Nguyên nhân: Streamlit chưa được cài trong môi trường đang chạy.

```powershell
python -m pip install streamlit
```

### Lỗi: No module named src

Chạy Streamlit từ thư mục gốc dự án:

```powershell
python -m streamlit run app/streamlit_app.py
```

### Lỗi: Model not found

Kiểm tra file:

```text
models/car_price_xgboost.joblib
```

### Lỗi: Expecting value: line 1 column 1

Lỗi có thể xảy ra khi chương trình cố đọc một file JSON rỗng hoặc không hợp lệ.

Nếu `src/predict.py` đã được cập nhật để lấy tên features từ `model.feature_names_in_`, không cần sử dụng `model_features.json`.

### Lỗi: Feature names mismatch

Kiểm tra dữ liệu đầu vào sau preprocessing.

Số features, tên cột và thứ tự cột phải giống với lúc model được huấn luyện.

### Cảnh báo phiên bản XGBoost

Model lưu bằng Joblib có thể phát sinh cảnh báo tương thích phiên bản khi tải bằng môi trường khác.

Nên sử dụng phiên bản thư viện tương thích với phiên bản lúc huấn luyện và chỉ tải model từ nguồn đáng tin cậy.

## 12. Hạn chế của dự án

### 12.1. Dataset

Dữ liệu CarDekho phản ánh thị trường ô tô tại Ấn Độ. Kết quả không thể áp dụng trực tiếp để định giá ô tô tại Việt Nam.

### 12.2. Feature Engineering

Đặc trưng `vehicle_age` đang sử dụng năm tham chiếu cố định **2026**.

Việc thay đổi năm tham chiếu có thể khiến dữ liệu đầu vào không tương thích với phân phối đặc trưng mà model đã học.

### 12.3. Unknown Categories

Nếu người dùng nhập hãng xe chưa xuất hiện trong dữ liệu train, model không có feature One-Hot Encoding riêng cho hãng đó.

Điều này có thể làm giảm độ chính xác.

### 12.4. Data Leakage

Notebook hiện thực hiện One-Hot Encoding trên toàn bộ dữ liệu trước khi chia train/test.

Trong các phiên bản tiếp theo, nên sử dụng Scikit-learn Pipeline để fit các phép tiền xử lý trên training set trước, sau đó áp dụng cho validation/test set.

### 12.5. Model Limitations

Model chưa sử dụng những yếu tố như:

- Lịch sử tai nạn
- Tình trạng động cơ
- Lịch sử bảo dưỡng
- Số lần sửa chữa
- Vị trí địa lý
- Biến động giá theo thời gian

Vì vậy, kết quả chỉ nên được xem là giá tham khảo.

## 13. Hướng phát triển

- Đánh giá lại mô hình với pipeline tránh data leakage.
- Thử nghiệm thêm các thuật toán như Random Forest và CatBoost.
- Cải thiện khả năng xử lý hãng xe và danh mục chưa từng xuất hiện.
- Bổ sung các yếu tố ảnh hưởng đến giá xe.
- Thêm trực quan hóa kết quả dự đoán trên Streamlit.
- Lưu lịch sử dự đoán trên Supabase.
- Tích hợp API bằng FastAPI nếu phát triển ứng dụng lớn hơn.
- Triển khai ứng dụng lên Streamlit Community Cloud.

Khi triển khai, cần đảm bảo server có thể truy cập file model. Nếu `.gitignore` loại trừ `.joblib`, phải cấu hình một phương thức phân phối model riêng.

## 14. Tài liệu tham khảo

- [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Supabase Documentation](https://supabase.com/docs)

---

**Trạng thái dự án:** Đã có notebook phân tích, tiền xử lý, model XGBoost đã huấn luyện và mã nguồn ứng dụng Streamlit. Cần tiếp tục kiểm thử toàn bộ quy trình trên máy đích và triển khai nếu muốn phát hành công khai.