# Dự đoán giá ô tô cũ với Machine Learning

## 1. Giới thiệu dự án

Thị trường ô tô đã qua sử dụng có sự đa dạng về hãng xe, năm sản xuất, số km đã đi, loại nhiên liệu và tình trạng sở hữu. Những đặc điểm này có thể liên quan đến giá bán, khiến việc ước lượng giá một chiếc xe cũ trở thành một bài toán đáng quan tâm.

Dự án Dự đoán giá ô tô cũ (Used Car Price Prediction) sử dụng bộ dữ liệu CarDekho để khám phá các yếu tố liên quan đến giá bán và xây dựng mô hình Machine Learning nhằm ước lượng giá xe dựa trên những thông tin có sẵn.

Dự án tập trung vào ba thuật toán hồi quy:
- Linear Regression: Mô hình hồi quy tuyến tính, được sử dụng làm mốc so sánh.
- Random Forest Regressor: Mô hình kết hợp nhiều cây quyết định theo phương pháp Bagging.
- XGBoost Regressor: Mô hình sử dụng Gradient Boosting để cải thiện khả năng dự đoán.
Thông qua quá trình khám phá dữ liệu, tiền xử lý, huấn luyện, tối ưu và đánh giá mô hình, dự án hướng đến việc so sánh hiệu quả của các thuật toán và lựa chọn phương pháp phù hợp cho bài toán dự đoán giá ô tô cũ.

1. Mục tiêu dự án
- Khám phá phân phối giá bán và mối quan hệ giữa các đặc trưng với giá xe.
- Chuẩn bị dữ liệu và xây dựng đặc trưng phù hợp cho bài toán hồi quy (Regression).
- Huấn luyện, so sánh và tối ưu Linear Regression, Random Forest và XGBoost.
- Đánh giá bằng MAE, RMSE và R² trên dữ liệu chưa được sử dụng để huấn luyện.
Biến mục tiêu (target): selling_price (đơn vị INR – rupee Ấn Độ).

2. Bộ dữ liệu
- Nguồn tham khảo: Vehicle Dataset from CarDekho – Kaggle
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
- Tệp dữ liệu: car_dataset.csv
- Kích thước ban đầu: 4.340 dòng × 8 cột
- Dữ liệu thiếu: Không phát hiện giá trị null trong 8 cột.
- Dòng trùng hoàn toàn: 763; sau khi loại bỏ còn 3.577 dòng.

Cột	Mô tả
name : Tên xe, bao gồm hãng, mẫu xe và phiên bản
year : Năm sản xuất
selling_price : Giá bán xe (INR), là biến mục tiêu
km_driven : Tổng số km xe đã đi
fuel : Loại nhiên liệu: Petrol, Diesel, CNG, LPG, Electric
seller_type : Loại người bán : Individual, Dealer, Trustmark Dealer
transmission : Hộp số: Manual, Automatic
owner : Thông tin sở hữu: First Owner, Second Owner, Third Owner, Fourth & Above Owner, Test Drive Car

## 3. Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| python-dotenv | Đọc cấu hình cục bộ từ file `.env` |
| Pandas, NumPy | Xử lý và phân tích dữ liệu |
| Matplotlib, Seaborn | Trực quan hóa dữ liệu |
| Scikit-learn | Tiền xử lý, xây dựng pipeline, huấn luyện và đánh giá mô hình |
| Microsoft SQL Server | Lưu trữ và truy vấn dữ liệu |
| SQLAlchemy, pyodbc | Kết nối Python với SQL Server |
| Jupyter Notebook | Khám phá dữ liệu và thử nghiệm mô hình |
| Joblib | Lưu và tải lại pipeline/mô hình |
| Git, GitHub | Quản lý phiên bản và mã nguồn |
| VS Code | Môi trường phát triển |

Danh sách công nghệ sẽ được cập nhật theo các thư viện thực tế được sử dụng trong dự án.

## 4. Cấu trúc thư mục

```text
salary-prediction/
├── data/
│   ├── raw/                              # Dữ liệu thô thu thập từ Adzuna API
│   └── processed/                        # Dữ liệu đã làm sạch/biến đổi nếu cần
│
├── notebooks/
│   ├── 01_data_understanding.ipynb        # Tìm hiểu cấu trúc và chất lượng dữ liệu
│   ├── 02_eda.ipynb                       # Phân tích khám phá dữ liệu
│   ├── 03_preprocessing.ipynb             # Thử nghiệm tiền xử lý
│   ├── 04_feature_engineering.ipynb       # Xây dựng và lựa chọn đặc trưng
│   ├── 05_model_experiments.ipynb         # Thử nghiệm các mô hình hồi quy
│   └── 06_evaluation.ipynb                # Đánh giá và phân tích lỗi mô hình
│
├
│
├── src/
│   ├── __init__.py
│   ├── config.py                          # Quản lý cấu hình và đường dẫn
│   ├── db.py                              # Kết nối SQL Server
│   ├── adzuna_client.py                   # Gọi API và xử lý phản hồi
│   ├── data_loader.py                     # Đọc và nạp dữ liệu
│   ├── data_validation.py                 # Kiểm tra chất lượng dữ liệu
│   ├── eda.py                             # Các hàm phân tích dữ liệu có thể tái sử dụng
│   ├── preprocessing.py                   # Tiền xử lý dữ liệu
│   ├── feature_engineering.py              # Xây dựng đặc trưng
│   ├── train.py                           # Huấn luyện và lưu mô hình
│   ├── evaluate.py                        # Đánh giá mô hình
│   └── predict.py                         # Thực hiện dự đoán
│
├── sql/
│   ├── 01_create_database.sql              # Tạo database
│   ├── 02_create_tables.sql                # Tạo các bảng dữ liệu
│   └── 03_quality_checks.sql               # Kiểm tra chất lượng dữ liệu bằng SQL
│
├── models/                                # Pipeline và mô hình đã huấn luyện
├── reports/                               # Biểu đồ, bảng so sánh và báo cáo
│
├── main.py                                # Điều phối pipeline chính
├── requirements.txt                       # Danh sách thư viện
├── .env.example                           # Mẫu cấu hình môi trường
├── .gitignore                             # Các tệp không đưa lên Git
└── README.md                              # Tài liệu dự án
```

Cấu trúc trên là cấu trúc mục tiêu. Các thư mục và file sẽ được hoàn thiện dần theo từng giai đoạn; không bắt buộc tất cả module phải có ngay từ đầu. Nếu cấu trúc dự án thực tế khác, hãy cập nhật README để phản ánh đúng mã nguồn hiện có.

## 5. Quy trình thực hiện dự án

5. Quy trình thực hiện dự án

Dự án được triển khai theo quy trình Machine Learning, từ khám phá dữ liệu đến xây dựng, tối ưu và đánh giá các mô hình dự đoán giá ô tô cũ.

5.1. Data Understanding (Tìm hiểu dữ liệu)

Trạng thái: Đã hoàn thành

Tìm hiểu nguồn gốc, cấu trúc và ý nghĩa các thuộc tính trong bộ dữ liệu CarDekho.

Xác định biến mục tiêu selling_price và các đặc trưng đầu vào.

Kiểm tra kiểu dữ liệu, giá trị thiếu và các bản ghi trùng lặp.

Thực hiện thống kê mô tả để hiểu đặc điểm ban đầu của dữ liệu.

Kết quả: Dataset gồm 4.340 bản ghi, 8 cột, không phát hiện giá trị thiếu và có 763 bản ghi trùng lặp hoàn toàn.

5.2. Exploratory Data Analysis – EDA (Phân tích khám phá dữ liệu)

Trạng thái: Đã thực hiện phân tích ban đầu

Univariate Analysis: Phân tích phân phối của biến mục tiêu selling_price và các đặc trưng.

Outlier Analysis: Phát hiện các giá trị ngoại lai trong selling_price và km_driven bằng Boxplot.

Bivariate Analysis: Phân tích mối quan hệ giữa các đặc trưng và giá bán ô tô.

Correlation Analysis: Sử dụng Pearson Correlation và Heatmap để đánh giá tương quan tuyến tính giữa các biến số.

Kết quả ban đầu: Giá bán có phân phối lệch phải, xuất hiện các giá trị ngoại lai và có sự khác biệt về giá giữa các nhóm xe. Các kết quả EDA được sử dụng để định hướng quá trình tiền xử lý và xây dựng mô hình.

5.3. Data Cleaning & Preprocessing (Làm sạch và tiền xử lý dữ liệu)

Trạng thái: Đã xử lý trùng lặp; các bước còn lại dự kiến triển khai

Loại bỏ 763 bản ghi trùng lặp hoàn toàn, còn lại 3.577 bản ghi.

Kiểm tra tính hợp lệ của dữ liệu và xem xét các giá trị ngoại lai.

Chia dữ liệu thành tập Train và Test trước các phép biến đổi cần học tham số.

Mã hóa các biến Categorical thành dạng số phù hợp với thuật toán.

Thực hiện Feature Engineering, tạo đặc trưng vehicle_age từ năm sản xuất.

Thử nghiệm Log Transformation cho các biến phù hợp khi cần thiết.

5.4. Model Training (Huấn luyện mô hình)

Trạng thái: Dự kiến thực hiện

Xây dựng và huấn luyện ba mô hình hồi quy:

Linear Regression: Mô hình hồi quy tuyến tính làm mốc so sánh.

Random Forest Regressor: Mô hình kết hợp nhiều cây quyết định theo phương pháp Bagging.

XGBoost Regressor: Mô hình sử dụng Gradient Boosting để cải thiện hiệu quả dự đoán.

Các mô hình được huấn luyện và đánh giá bằng cùng một quy trình nhằm đảm bảo tính nhất quán khi so sánh.

5.5. Model Optimization (Tối ưu mô hình)

Trạng thái: Dự kiến thực hiện

So sánh hiệu quả ban đầu của các mô hình bằng Cross-Validation.

Lựa chọn các mô hình có kết quả triển vọng để tiếp tục tối ưu.

Thử nghiệm Log Transformation nhằm cải thiện khả năng dự đoán.

Sử dụng Hyperparameter Tuning để tìm bộ siêu tham số phù hợp.

So sánh kết quả trước và sau tối ưu để lựa chọn mô hình tốt nhất.

5.6. Model Evaluation (Đánh giá mô hình)

Trạng thái: Dự kiến thực hiện

Đánh giá mô hình được lựa chọn trên tập Test độc lập bằng các chỉ số:

MAE (Mean Absolute Error): Sai số tuyệt đối trung bình.

RMSE (Root Mean Squared Error): Căn bậc hai của sai số bình phương trung bình.

R² (Coefficient of Determination): Hệ số xác định.

Bên cạnh đó, sử dụng các biểu đồ Actual vs Predicted, Residual Plot và Feature Importance để phân tích kết quả dự đoán và vai trò của các đặc trưng.

Kết quả mong muốn: Xác định mô hình có khả năng dự đoán giá ô tô cũ hiệu quả, tổng quát hóa tốt trên dữ liệu chưa từng thấy và phù hợp với mục tiêu của dự án.