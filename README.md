# 🩺 Diabetes Prediction — Dự án dự đoán bệnh tiểu đường bằng Machine Learning

## 1. Giới thiệu dự án

**Diabetes Prediction** là dự án xây dựng hệ thống dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các đặc trưng của dữ liệu bệnh nhân, sử dụng các thuật toán Machine Learning.

Dự án được xây dựng theo quy trình phát triển một hệ thống Machine Learning từ đầu đến cuối (*End-to-End Machine Learning Pipeline*), bao gồm thu thập dữ liệu, lưu trữ dữ liệu, phân tích khám phá, tiền xử lý, huấn luyện mô hình, đánh giá và triển khai ứng dụng dự đoán.

Bên cạnh việc tìm hiểu các thuật toán Machine Learning, dự án hướng đến việc rèn luyện kỹ năng lập trình Python theo mô-đun, quản lý dữ liệu bằng SQL Server và xây dựng quy trình có thể tái sử dụng.

**Lưu ý:** Dự án phục vụ mục đích học tập và nghiên cứu. Kết quả dự đoán không thay thế cho việc thăm khám, tư vấn hoặc chẩn đoán của nhân viên y tế.

## 2. Mục tiêu dự án

- Tự động tải dữ liệu từ Kaggle thông qua Kaggle API.
- Sử dụng Microsoft SQL Server để lưu trữ và truy vấn dữ liệu.
- Thực hiện Data Understanding và Exploratory Data Analysis (EDA).
- Phát hiện dữ liệu thiếu, dữ liệu trùng lặp, giá trị bất thường và mất cân bằng lớp.
- Xây dựng quy trình tiền xử lý dữ liệu phù hợp với bài toán phân loại.
- Thử nghiệm và so sánh nhiều thuật toán Machine Learning.
- Đánh giá mô hình bằng các chỉ số phù hợp.
- Lưu mô hình đã lựa chọn để tái sử dụng cho việc dự đoán.
- Xây dựng giao diện dự đoán bằng Streamlit.
- Tổ chức mã nguồn rõ ràng, có khả năng mở rộng và quản lý bằng Git/GitHub.

## 3. Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| Kaggle API | Tự động tải dữ liệu |
| Pandas, NumPy | Xử lý và phân tích dữ liệu |
| Matplotlib, Seaborn | Trực quan hóa dữ liệu |
| Scikit-learn | Tiền xử lý, huấn luyện và đánh giá mô hình |
| Microsoft SQL Server | Lưu trữ và truy vấn dữ liệu |
| SQLAlchemy, pyodbc | Kết nối Python với SQL Server |
| Jupyter Notebook | Khám phá dữ liệu và thử nghiệm mô hình |
| Joblib | Lưu và tải lại mô hình |
| Streamlit | Xây dựng giao diện dự đoán |
| Git, GitHub | Quản lý phiên bản và mã nguồn |
| VS Code | Môi trường phát triển |

## 4. Cấu trúc thư mục

```text
diabetes-prediction/
├── data/
│   ├── raw/                         # Dữ liệu gốc tải từ Kaggle
│   └── processed/                   # Dữ liệu đã xử lý nếu cần
│
├── notebooks/
│   ├── 01_data_understanding.ipynb  # Tìm hiểu cấu trúc dữ liệu
│   ├── 02_eda.ipynb                 # Phân tích khám phá dữ liệu
│   ├── 03_preprocessing.ipynb       # Thử nghiệm tiền xử lý
│   ├── 04_model_experiments.ipynb   # Thử nghiệm các mô hình
│   └── 05_evaluation_explainability.ipynb
│
├── scripts/
│   └── download_kaggle_data.py      # Tải dữ liệu từ Kaggle
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Quản lý cấu hình
│   ├── db.py                       # Kết nối SQL Server
│   ├── data_loader.py              # Đọc và nạp dữ liệu
│   ├── data_validation.py          # Kiểm tra chất lượng dữ liệu
│   ├── eda.py                      # Các hàm phân tích dữ liệu
│   ├── preprocessing.py            # Tiền xử lý dữ liệu
│   ├── train.py                    # Huấn luyện mô hình
│   ├── evaluate.py                 # Đánh giá mô hình
│   └── predict.py                  # Thực hiện dự đoán
│
├── sql/
│   ├── 01_create_database.sql      # Tạo database
│   ├── 02_create_tables.sql        # Tạo các bảng dữ liệu
│   └── 03_quality_checks.sql       # Kiểm tra dữ liệu bằng SQL
│
├── models/                         # Mô hình và pipeline đã huấn luyện
├── reports/                        # Biểu đồ và báo cáo kết quả
│
├── app/
│   └── streamlit_app.py            # Giao diện ứng dụng
│
├── main.py                         # Điều phối pipeline chính
├── requirements.txt                # Danh sách thư viện
├── .env.example                    # Mẫu cấu hình môi trường
├── .gitignore                      # Các tệp không đưa lên Git
└── README.md                       # Tài liệu dự án
```

Các thư mục và file sẽ được hoàn thiện dần theo từng giai đoạn. Không bắt buộc tất cả các module phải có ngay từ đầu.

## 5. Quy trình thực hiện dự án

### Giai đoạn 1 — Thu thập dữ liệu

**Mục tiêu:** Tự động lấy dữ liệu và xác định chính xác cấu trúc dataset.

Các công việc:

1. Lựa chọn dataset phù hợp trên Kaggle.
2. Cấu hình Kaggle API.
3. Sử dụng `scripts/download_kaggle_data.py` để tải và giải nén dữ liệu.
4. Lưu dữ liệu gốc trong `data/raw/`.
5. Kiểm tra tên file, số lượng bản ghi, số lượng đặc trưng, kiểu dữ liệu và biến mục tiêu.

**Kết quả đầu ra:** Dataset được tải về và sẵn sàng cho bước phân tích.

### Giai đoạn 2 — Thiết lập SQL Server

**Mục tiêu:** Xây dựng nơi lưu trữ và truy vấn dữ liệu.

Các công việc:

1. Tạo database cho dự án.
2. Thiết kế bảng phù hợp với cấu trúc dataset thực tế.
3. Nạp dữ liệu vào SQL Server.
4. Kiểm tra số lượng bản ghi, giá trị thiếu và các vấn đề chất lượng dữ liệu.
5. Thực hiện các truy vấn thống kê cần thiết.
6. Kết nối SQL Server với Python bằng SQLAlchemy và pyodbc.

**Kết quả đầu ra:** Database hoạt động và có thể được truy cập từ Python.

### Giai đoạn 3 — Data Understanding và EDA

**Mục tiêu:** Hiểu dữ liệu trước khi xây dựng mô hình.

Thực hiện trong `01_data_understanding.ipynb` và `02_eda.ipynb`.

Các công việc:

- Xác định số lượng bản ghi và đặc trưng.
- Phân biệt biến số, biến phân loại và biến mục tiêu.
- Kiểm tra giá trị thiếu và bản ghi trùng lặp.
- Phân tích thống kê mô tả.
- Khảo sát phân phối của các đặc trưng.
- Kiểm tra giá trị bất thường và tính hợp lệ của dữ liệu.
- Phân tích phân phối giữa các lớp của biến mục tiêu.
- Trực quan hóa mối quan hệ giữa đặc trưng và biến mục tiêu.

**Kết quả đầu ra:** Báo cáo EDA, biểu đồ và danh sách các vấn đề cần xử lý.

### Giai đoạn 4 — Tiền xử lý dữ liệu

**Mục tiêu:** Chuẩn bị dữ liệu phù hợp với các thuật toán Machine Learning.

Thực hiện trong `03_preprocessing.ipynb`, sau đó triển khai thành `src/preprocessing.py`.

Các công việc có thể bao gồm:

- Xử lý dữ liệu thiếu.
- Kiểm tra và xử lý các giá trị không hợp lệ.
- Xử lý bản ghi trùng lặp khi có cơ sở phù hợp.
- Mã hóa các biến phân loại nếu dataset có.
- Chuẩn hóa đặc trưng khi thuật toán yêu cầu.
- Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
- Xem xét mất cân bằng lớp nếu có.

**Nguyên tắc quan trọng:** Tránh rò rỉ dữ liệu (*data leakage*). Các bước học tham số từ dữ liệu, chẳng hạn như điền giá trị thiếu hoặc chuẩn hóa, phải được khớp trên tập huấn luyện và sau đó áp dụng cho các tập còn lại.

**Kết quả đầu ra:** Quy trình tiền xử lý có thể tái sử dụng khi huấn luyện và dự đoán.

### Giai đoạn 5 — Huấn luyện và thử nghiệm mô hình

**Mục tiêu:** So sánh các thuật toán và tìm mô hình phù hợp với dữ liệu.

Thực hiện trong `04_model_experiments.ipynb`.

Các thuật toán dự kiến khảo sát:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Danh sách trên là các ứng viên ban đầu, không phải yêu cầu phải sử dụng tất cả. Việc lựa chọn cuối cùng sẽ dựa trên kết quả thực nghiệm, độ phức tạp, khả năng giải thích và yêu cầu của dự án.

Các công việc:

1. Xây dựng mô hình baseline.
2. Huấn luyện từng thuật toán.
3. So sánh kết quả trên cùng một cách chia dữ liệu.
4. Điều chỉnh tham số khi cần.
5. Lựa chọn mô hình phù hợp dựa trên nhiều tiêu chí.

**Kết quả đầu ra:** Bảng so sánh các mô hình và mô hình ứng viên được lựa chọn.

### Giai đoạn 6 — Đánh giá và giải thích mô hình

**Mục tiêu:** Đánh giá hiệu quả và hiểu rõ những hạn chế của mô hình.

Thực hiện trong `05_evaluation_explainability.ipynb`.

Các chỉ số có thể sử dụng:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC, khi phù hợp

Đối với bài toán sức khỏe, không nên chỉ dựa vào Accuracy. Cần xem xét Recall của lớp dương tính, số lượng trường hợp âm tính giả (*false negatives*), độ tin cậy của nhãn và khả năng khái quát hóa của mô hình.

Có thể nghiên cứu thêm các phương pháp giải thích như feature importance hoặc SHAP nếu phù hợp.

**Kết quả đầu ra:** Báo cáo đánh giá, phân tích lỗi và lý do lựa chọn mô hình.

### Giai đoạn 7 — Xây dựng Python Pipeline

**Mục tiêu:** Chuyển các thử nghiệm đã ổn định thành mã nguồn có thể tái sử dụng.

| File | Chức năng |
|---|---|
| `config.py` | Quản lý cấu hình và các đường dẫn |
| `db.py` | Kết nối và thao tác với SQL Server |
| `data_loader.py` | Đọc hoặc nạp dữ liệu |
| `data_validation.py` | Kiểm tra cấu trúc và chất lượng dữ liệu |
| `eda.py` | Tái sử dụng các hàm phân tích |
| `preprocessing.py` | Xây dựng quy trình tiền xử lý |
| `train.py` | Huấn luyện và lưu mô hình |
| `evaluate.py` | Tính toán chỉ số đánh giá |
| `predict.py` | Tải pipeline đã huấn luyện và dự đoán |
| `main.py` | Điều phối quy trình thực thi |

Notebook chủ yếu phục vụ học tập, khám phá và thử nghiệm. Quy trình chạy chính thức sẽ gọi các module Python, thay vì phụ thuộc vào việc chạy thủ công từng notebook.

Pipeline tiền xử lý và mô hình nên được lưu cùng nhau để đảm bảo dữ liệu đầu vào được xử lý nhất quán khi dự đoán.

### Giai đoạn 8 — Xây dựng ứng dụng dự đoán

**Mục tiêu:** Cho phép người dùng nhập thông tin đầu vào và xem kết quả dự đoán.

Sử dụng Streamlit để xây dựng giao diện đơn giản.

Các chức năng dự kiến:

- Nhập các đặc trưng theo dataset đã lựa chọn.
- Kiểm tra tính hợp lệ của dữ liệu đầu vào.
- Tải pipeline Machine Learning đã lưu.
- Hiển thị kết quả phân loại và xác suất dự đoán nếu mô hình hỗ trợ.
- Giải thích các giới hạn của kết quả.
- Ghi lại lịch sử dự đoán trong SQL Server nếu cần.

Ứng dụng không được trình bày kết quả như một chẩn đoán y khoa xác nhận.

## 6. Cài đặt môi trường

### Yêu cầu

- Python.
- Visual Studio Code.
- Microsoft SQL Server.
- SQL Server ODBC Driver tương thích.
- Tài khoản Kaggle và quyền truy cập dataset.
- Git.

### Bước 1 — Tạo môi trường ảo

Mở terminal tại thư mục gốc dự án và chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn việc kích hoạt môi trường, có thể sử dụng terminal phù hợp hoặc cấu hình chính sách thực thi theo hướng dẫn dành cho Windows.

### Bước 2 — Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

File `requirements.txt` sẽ được cập nhật theo các thư viện thực tế mà dự án sử dụng.

### Bước 3 — Cấu hình Kaggle API

1. Đăng nhập tài khoản Kaggle.
2. Mở phần cài đặt tài khoản và tạo API token theo hướng dẫn hiện hành.
3. Cấu hình thông tin xác thực theo tài liệu Kaggle API.
4. Không đưa token hoặc thông tin xác thực lên GitHub.

### Bước 4 — Tải dữ liệu

Sau khi cấu hình Kaggle API và dataset ID:

```powershell
python scripts/download_kaggle_data.py
```

Dữ liệu sẽ được lưu vào `data/raw/` theo cấu hình của script.

### Bước 5 — Thiết lập SQL Server

Thực hiện các script SQL theo thứ tự:

1. `sql/01_create_database.sql`
2. `sql/02_create_tables.sql`
3. `sql/03_quality_checks.sql`

Tên database, tên bảng và cấu trúc cột sẽ được điều chỉnh theo dataset thực tế.

### Bước 6 — Chạy pipeline

Sau khi hoàn thiện các module:

```powershell
python main.py
```

### Bước 7 — Chạy ứng dụng

Sau khi huấn luyện và lưu mô hình:

```powershell
streamlit run app/streamlit_app.py
```

Các lệnh trên mô tả cách sử dụng dự kiến. Pipeline và giao diện chỉ hoạt động hoàn chỉnh sau khi các file tương ứng được triển khai.

## 7. Quản lý dữ liệu và bảo mật

- Giữ nguyên dữ liệu gốc trong `data/raw/` khi có thể.
- Không đưa API token, mật khẩu SQL Server hoặc thông tin bí mật vào mã nguồn.
- Sử dụng `.env` cho cấu hình cục bộ và loại file này khỏi Git.
- Duy trì `.env.example` chỉ chứa các tên biến và giá trị mẫu không nhạy cảm.
- Hạn chế đưa dataset lớn, file mô hình và dữ liệu nhạy cảm lên GitHub.
- Kiểm tra giấy phép và điều kiện sử dụng dataset trước khi chia sẻ.
- Ghi lại cấu hình thí nghiệm, random seed và các bước tiền xử lý quan trọng.

## 8. Hạn chế của dự án

- Kết quả phụ thuộc vào chất lượng, kích thước và tính đại diện của dataset.
- Dữ liệu công khai có thể không phản ánh đầy đủ các nhóm dân số hoặc điều kiện thực tế.
- Hiệu suất tốt trên tập kiểm tra không đồng nghĩa mô hình đã được xác nhận về mặt lâm sàng.
- Mô hình có thể mắc sai sót, đặc biệt với những trường hợp dữ liệu ít gặp.
- Kết quả chỉ có ý nghĩa trong phạm vi dữ liệu và phương pháp đánh giá đã thực hiện.

## 9. Tiến độ dự án

- [ ] Lựa chọn dataset Kaggle và xác minh giấy phép.
- [ ] Hoàn thiện script tải dữ liệu bằng Kaggle API.
- [ ] Thiết lập SQL Server và cấu trúc bảng.
- [ ] Hoàn thành Data Understanding và EDA.
- [ ] Xây dựng quy trình tiền xử lý.
- [ ] Huấn luyện các mô hình baseline.
- [ ] So sánh và lựa chọn mô hình.
- [ ] Đánh giá và giải thích kết quả.
- [ ] Hoàn thiện các module Python.
- [ ] Tích hợp pipeline trong `main.py`.
- [ ] Xây dựng giao diện Streamlit.
- [ ] Hoàn thiện tài liệu và báo cáo dự án.

## 10. Nguồn dữ liệu và giấy phép

Thông tin về dataset, tác giả, đường dẫn Kaggle, giấy phép và yêu cầu trích dẫn sẽ được bổ sung sau khi dataset chính thức được lựa chọn.

---

**Trạng thái dự án:** Đang trong giai đoạn thiết kế cấu trúc và chuẩn bị môi trường.

**Định hướng:** Xây dựng một dự án Machine Learning có quy trình rõ ràng, mã nguồn dễ bảo trì và có thể mở rộng thành ứng dụng dự đoán hoàn chỉnh.
