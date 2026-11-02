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

### Giai đoạn 8 — Xây dựng Python Pipeline

**Mục tiêu:** Chuyển các thử nghiệm ổn định thành mã nguồn có thể tái sử dụng và chạy theo quy trình thống nhất.

| File | Chức năng |
|---|---|
| `config.py` | Quản lý cấu hình và đường dẫn |
| `adzuna_client.py` | Gọi Adzuna API và xử lý phản hồi |
| `db.py` | Kết nối và thao tác với SQL Server |
| `data_loader.py` | Đọc hoặc nạp dữ liệu |
| `data_validation.py` | Kiểm tra cấu trúc và chất lượng dữ liệu |
| `eda.py` | Tái sử dụng các hàm phân tích |
| `preprocessing.py` | Xử lý dữ liệu đầu vào |
| `feature_engineering.py` | Xây dựng và biến đổi đặc trưng |
| `train.py` | Huấn luyện và lưu pipeline/mô hình |
| `evaluate.py` | Tính toán chỉ số đánh giá |
| `predict.py` | Tải pipeline đã huấn luyện và dự đoán |
| `main.py` | Điều phối quy trình thực thi |

Notebook chủ yếu phục vụ học tập, khám phá và thử nghiệm. Các module Python sẽ đảm nhiệm những bước ổn định để có thể chạy lại quy trình mà không cần thao tác thủ công từng notebook.

Pipeline tiền xử lý và mô hình nên được lưu cùng nhau để bảo đảm dữ liệu đầu vào được biến đổi nhất quán lúc huấn luyện và dự đoán. Các bước chuẩn hóa dữ liệu, xây dựng đặc trưng và thứ tự cột phải giống nhau giữa hai giai đoạn.

Ở giai đoạn đầu, có thể chạy các module riêng lẻ. Chỉ tích hợp toàn bộ vào `main.py` sau khi các thành phần cần thiết đã được kiểm tra.

**Kết quả đầu ra:** Các module Python có thể tái sử dụng và pipeline dự đoán có thể được tải lại.

### Giai đoạn 9 — Dự đoán thử nghiệm và hướng phát triển ứng dụng

**Mục tiêu:** Kiểm tra khả năng sử dụng pipeline đã huấn luyện với dữ liệu đầu vào mới.

Trong giai đoạn tập trung vào Machine Learning, có thể kiểm tra bằng `src/predict.py` hoặc một notebook riêng.

Các chức năng dự kiến:

- Nhận các đặc trưng đầu vào phù hợp với schema mô hình.
- Kiểm tra kiểu dữ liệu và tính hợp lệ của đầu vào.
- Áp dụng đúng pipeline tiền xử lý đã lưu.
- Tạo dự đoán mức lương và trả về kết quả theo đơn vị mục tiêu đã xác nhận.
- Kiểm tra trường hợp đầu vào thiếu đặc trưng hoặc chứa giá trị chưa từng xuất hiện.
- Ghi lại ví dụ kiểm thử để xác minh khả năng tái sử dụng mô hình.

**Streamlit chưa nằm trong phạm vi triển khai hiện tại.** Sau khi quy trình dữ liệu, huấn luyện và dự đoán đã ổn định, có thể cân nhắc phát triển giao diện để người dùng nhập thông tin công việc và xem mức lương ước tính.

**Kết quả đầu ra:** Module dự đoán có thể chạy thử bằng dữ liệu mới; giao diện người dùng được xem là hướng phát triển sau.

## 6. Cài đặt môi trường

### Yêu cầu

- Python phiên bản tương thích với các thư viện của dự án.
- Visual Studio Code.
- Microsoft SQL Server.
- SQL Server ODBC Driver tương thích.
- Tài khoản và thông tin xác thực Adzuna API.
- Git.
- Kết nối Internet để gọi API trong giai đoạn thu thập dữ liệu.

### Bước 1 — Tạo môi trường ảo

Mở terminal tại thư mục gốc dự án và chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn việc kích hoạt môi trường, có thể sử dụng terminal phù hợp hoặc cấu hình chính sách thực thi theo hướng dẫn dành cho Windows.

### Bước 2 — Cài đặt thư viện

Sau khi kích hoạt môi trường ảo:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` cần được cập nhật theo các thư viện thực tế mà dự án sử dụng. Các gói thường cần cân nhắc gồm `requests`, `python-dotenv`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `sqlalchemy`, `pyodbc`, `jupyter` và `joblib`.

### Bước 3 — Cấu hình Adzuna API

1. Đăng ký hoặc đăng nhập tài khoản Adzuna và đăng ký API theo hướng dẫn chính thức.
2. Lấy thông tin xác thực được cấp, chẳng hạn `APP_ID` và `APP_KEY`.
3. Tạo file `.env` tại thư mục gốc dự án.
4. Thêm thông tin xác thực vào `.env` theo tên biến mà chương trình sử dụng.
5. Đảm bảo `.env` được đưa vào `.gitignore`.
6. Chỉ đưa tên biến và giá trị minh họa không nhạy cảm vào `.env.example`.

Ví dụ cấu trúc `.env.example`:

```dotenv
ADZUNA_APP_ID=your_app_id_here
ADZUNA_APP_KEY=your_app_key_here
```

Tên biến trong ví dụ cần khớp với cách đọc cấu hình trong mã nguồn. Không commit API key thật lên GitHub, không đưa key vào notebook đã lưu và không chia sẻ key trong log hoặc ảnh chụp màn hình.

### Bước 4 — Thu thập dữ liệu

Sau khi cấu hình API và hoàn thiện script:

```powershell
python scripts/fetch_adzuna_data.py
```

Script sẽ gọi API theo tham số đã cấu hình và lưu dữ liệu vào `data/raw/`. Tên file đầu ra, quốc gia, từ khóa, số trang và các trường được thu thập cần được xác định trong cấu hình thực tế.

Cần tuân thủ giới hạn truy vấn, điều khoản sử dụng và quy định về dữ liệu của Adzuna. Nếu API trả về lỗi hoặc hết hạn mức, script nên thông báo rõ thay vì âm thầm tạo ra một file dữ liệu có vẻ hợp lệ nhưng không đầy đủ.

### Bước 5 — Thiết lập SQL Server

Thực hiện các script SQL theo thứ tự:

1. `sql/01_create_database.sql`
2. `sql/02_create_tables.sql`
3. `sql/03_quality_checks.sql`

Tên database, bảng, kiểu dữ liệu và cột kết nối sẽ được điều chỉnh theo schema thực tế. Cấu hình kết nối nên được đặt trong biến môi trường thay vì ghi mật khẩu trực tiếp vào mã nguồn.

### Bước 6 — Thực hiện phân tích và huấn luyện

Mở các notebook trong VS Code và chạy theo thứ tự:

1. `01_data_understanding.ipynb`
2. `02_eda.ipynb`
3. `03_preprocessing.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_model_experiments.ipynb`
6. `06_evaluation.ipynb`

Các notebook chỉ hoạt động đầy đủ khi dữ liệu đã được thu thập và các bước cần thiết đã được triển khai.

### Bước 7 — Chạy pipeline Python

Sau khi các module tương ứng được hoàn thiện:

```powershell
python main.py
```

Lệnh trên là cách chạy dự kiến. Các bước cụ thể mà `main.py` thực hiện cần được cập nhật theo cách triển khai thực tế của dự án.

### Bước 8 — Chạy dự đoán thử nghiệm

Khi đã huấn luyện và lưu pipeline, sử dụng module dự đoán theo giao diện được triển khai trong mã nguồn. Ví dụ lệnh dự kiến:

```powershell
python -m src.predict
```

Lệnh này chỉ áp dụng nếu `src/predict.py` có điểm vào chương trình tương ứng. Nếu module chỉ cung cấp hàm dự đoán, cần gọi hàm đó từ một script kiểm thử hoặc notebook.

## 7. Quản lý dữ liệu và bảo mật

- Giữ dữ liệu thô trong `data/raw/` để có thể kiểm tra và tái xử lý.
- Không đưa Adzuna API key, mật khẩu SQL Server hoặc thông tin xác thực vào mã nguồn.
- Sử dụng `.env` cho cấu hình cục bộ và loại file này khỏi Git.
- Duy trì `.env.example` chỉ chứa tên biến và giá trị mẫu không nhạy cảm.
- Hạn chế đưa dữ liệu lớn, thông tin không được phép chia sẻ và file mô hình lớn lên GitHub.
- Kiểm tra điều khoản API, giấy phép và điều kiện sử dụng dữ liệu trước khi phân phối dữ liệu thu thập được.
- Tôn trọng giới hạn truy vấn, phân trang và các điều kiện sử dụng do nhà cung cấp quy định.
- Ghi lại thời điểm thu thập, tham số truy vấn, số lượng bản ghi và các bước biến đổi dữ liệu.
- Không tự động nạp lại cùng một tin tuyển dụng thành nhiều bản ghi khác nhau nếu có thể nhận diện bản ghi trùng.
- Lưu cấu hình thí nghiệm, random seed, phiên bản thư viện và các bước tiền xử lý quan trọng.
- Kiểm tra đơn vị tiền tệ và chu kỳ trả lương trước khi kết hợp dữ liệu hoặc diễn giải kết quả.
- Không sử dụng các trường chứa trực tiếp thông tin nhãn mục tiêu làm đặc trưng đầu vào theo cách gây rò rỉ dữ liệu.

## 8. Hạn chế của dự án

- Độ phủ dữ liệu lương có thể thấp hơn độ phủ dữ liệu tin tuyển dụng; những tin không có đủ `salary_min` và `salary_max` không thể tạo nhãn mục tiêu theo công thức đã chọn.
- Dữ liệu thu thập qua API phụ thuộc vào quốc gia, từ khóa, địa điểm, thời điểm truy vấn, giới hạn API và các trường mà nhà cung cấp trả về.
- Mức lương được đăng có thể khác với mức lương thực tế mà ứng viên nhận được.
- Dữ liệu có thể trộn lẫn các loại tiền tệ, chu kỳ trả lương hoặc cách ghi lương khác nhau; việc xử lý không đúng sẽ làm giảm ý nghĩa của mô hình.
- Tin tuyển dụng trùng lặp, thông tin thiếu, giá trị bất thường và mô tả không nhất quán có thể ảnh hưởng đến chất lượng mô hình.
- Thị trường lao động thay đổi theo thời gian, địa điểm, ngành nghề và kinh nghiệm; mô hình cần được đánh giá lại khi dữ liệu thay đổi.
- Mô hình có thể dự đoán kém với các chức danh hiếm hoặc nhóm dữ liệu ít mẫu.
- Kết quả đánh giá trên tập kiểm tra không bảo đảm hiệu quả tương tự trên mọi tin tuyển dụng mới.
- MAE và RMSE phụ thuộc vào đơn vị của biến mục tiêu; không thể diễn giải đúng nếu chưa xác nhận tiền tệ và chu kỳ trả lương.
- Dự án phục vụ mục đích học tập và thử nghiệm Machine Learning; kết quả không phải tư vấn nghề nghiệp hoặc cam kết lương.

## 9. Tiến độ dự án

- [ ] Thiết lập quyền truy cập Adzuna API và kiểm tra tài liệu API.
- [ ] Hoàn thiện script thu thập dữ liệu và xử lý phân trang/lỗi.
- [ ] Lưu dữ liệu thô và ghi nhận thông tin truy vấn.
- [ ] Xác minh schema thực tế và các trường dữ liệu có sẵn.
- [ ] Thiết lập SQL Server và cấu trúc bảng.
- [ ] Nạp dữ liệu và kiểm tra chất lượng bằng SQL/Python.
- [ ] Hoàn thành Data Understanding và EDA.
- [ ] Xác minh đơn vị tiền tệ, chu kỳ trả lương và quy tắc tạo `salary`.
- [ ] Xây dựng quy trình tiền xử lý.
- [ ] Thử nghiệm Feature Engineering.
- [ ] Huấn luyện baseline cho bài toán Regression.
- [ ] Thử nghiệm và so sánh các mô hình hồi quy.
- [ ] Đánh giá bằng MAE, RMSE và \(R^2\).
- [ ] Phân tích sai số và lựa chọn mô hình.
- [ ] Hoàn thiện các module Python.
- [ ] Tích hợp pipeline trong `main.py`.
- [ ] Kiểm thử dự đoán với dữ liệu mới.
- [ ] Hoàn thiện README và báo cáo kết quả thực nghiệm.
- [ ] Xem xét phát triển giao diện Streamlit ở giai đoạn sau.

## 10. Nguồn dữ liệu và tài liệu tham khảo

**Nguồn dữ liệu chính:** Adzuna API.

- Trang chủ Adzuna: https://www.adzuna.com/
- Tài liệu API: https://developer.adzuna.com/

Thông tin cụ thể về endpoint, quốc gia, tham số truy vấn, giới hạn sử dụng, schema phản hồi và điều khoản dữ liệu cần được xác nhận theo tài liệu API và cấu hình tài khoản đang sử dụng.

Thông tin về thời gian thu thập, các truy vấn đã dùng, phạm vi dữ liệu và số lượng bản ghi thực tế sẽ được bổ sung sau khi hoàn thành bước thu thập. Không nên ghi số lượng bản ghi, chất lượng mô hình hoặc kết quả đánh giá khi chưa có thực nghiệm tương ứng.

---

**Trạng thái dự án:** Đang xây dựng quy trình thu thập dữ liệu và chuẩn bị pipeline Machine Learning.

**Định hướng:** Xây dựng một dự án Regression có workflow rõ ràng, quản lý dữ liệu bằng SQL Server, mã nguồn Python dễ bảo trì và có thể mở rộng thành ứng dụng dự đoán lương trong tương lai.