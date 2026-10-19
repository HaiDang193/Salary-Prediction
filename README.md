# 💼 Salary Prediction — Dự án dự đoán mức lương bằng Machine Learning

## 1. Giới thiệu dự án

**Salary Prediction** là dự án xây dựng mô hình Machine Learning để dự đoán mức lương ước tính của một tin tuyển dụng dựa trên các đặc trưng của công việc, sử dụng dữ liệu thu thập thông qua **Adzuna API**.

Dự án được phát triển theo quy trình **End-to-End Machine Learning Pipeline**, bao gồm thu thập dữ liệu từ API, lưu trữ dữ liệu bằng SQL Server, tìm hiểu và phân tích dữ liệu, tiền xử lý, xây dựng đặc trưng, huấn luyện mô hình hồi quy, đánh giá mô hình và đóng gói các thành phần Python để có thể tái sử dụng.

Biến mục tiêu dự kiến là `salary`, được tính từ `salary_min` và `salary_max` khi cả hai giá trị đều hợp lệ:

\[
salary = \frac{salary\_min + salary\_max}{2}
\]
 
Các bản ghi không có đủ thông tin để tính mức lương trung bình sẽ được kiểm tra và xử lý theo quy tắc dữ liệu đã xác định. Không tự động coi giá trị thiếu là 0 hoặc tự suy đoán mức lương nếu không có cơ sở.

Bên cạnh việc tìm hiểu các thuật toán Machine Learning, dự án hướng đến rèn luyện Python theo mô-đun, truy vấn dữ liệu bằng SQL Server, xây dựng quy trình thực nghiệm có thể lặp lại và quản lý mã nguồn bằng Git/GitHub.

**Lưu ý:** Kết quả dự đoán là mức lương ước tính dựa trên dữ liệu và phạm vi huấn luyện. Kết quả không phải cam kết về mức lương thực tế của một vị trí tuyển dụng.

## 2. Mục tiêu dự án

- Thu thập dữ liệu tin tuyển dụng thông qua Adzuna API.
- Quản lý API key và cấu hình truy vấn an toàn bằng biến môi trường.
- Xây dựng quy trình lấy dữ liệu có hỗ trợ phân trang, kiểm tra phản hồi API và xử lý lỗi.
- Sử dụng Microsoft SQL Server để lưu trữ, truy vấn và kiểm tra dữ liệu.
- Thực hiện Data Understanding và Exploratory Data Analysis (EDA).
- Kiểm tra dữ liệu thiếu, dữ liệu trùng lặp, giá trị bất thường và tính hợp lệ của mức lương.
- Xây dựng biến mục tiêu `salary` từ `salary_min` và `salary_max` khi đủ dữ liệu.
- Tiền xử lý dữ liệu số, dữ liệu phân loại và dữ liệu văn bản nếu các trường này có trong dữ liệu thu thập được.
- Xây dựng và so sánh các mô hình Machine Learning cho bài toán Regression.
- Đánh giá mô hình bằng các chỉ số phù hợp như MAE, RMSE và \(R^2\).
- Lưu pipeline tiền xử lý và mô hình đã lựa chọn để tái sử dụng.
- Tổ chức mã nguồn theo mô-đun, có khả năng mở rộng và quản lý bằng Git/GitHub.
- Tập trung hoàn thiện quy trình Machine Learning trước khi xem xét xây dựng giao diện người dùng.

## 3. Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| Adzuna API | Thu thập dữ liệu tin tuyển dụng và thông tin lương được API cung cấp |
| Requests | Gửi HTTP request đến API |
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
├── scripts/
│   └── fetch_adzuna_data.py               # Thu thập dữ liệu từ Adzuna API
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

### Giai đoạn 1 — Thu thập dữ liệu từ Adzuna API

**Mục tiêu:** Xây dựng quy trình thu thập tin tuyển dụng có thể chạy lại và kiểm tra được nguồn dữ liệu.

Các công việc:

1. Đăng ký và thiết lập quyền truy cập Adzuna API theo tài liệu chính thức.
2. Lấy thông tin xác thực cần thiết, thường gồm `APP_ID` và `APP_KEY`, theo cấu hình API được cấp.
3. Lưu thông tin xác thực trong file `.env` cục bộ; không ghi trực tiếp vào mã nguồn.
4. Xác định quốc gia, từ khóa tìm kiếm, địa điểm, số trang và các tham số truy vấn phù hợp.
5. Viết `scripts/fetch_adzuna_data.py` để gửi request, kiểm tra mã trạng thái HTTP và xử lý lỗi.
6. Thu thập dữ liệu theo phân trang trong phạm vi và giới hạn API cho phép.
7. Lưu phản hồi dữ liệu thô vào `data/raw/` để có thể kiểm tra và tái xử lý.
8. Ghi lại thời điểm thu thập, tham số truy vấn và số lượng bản ghi nhận được.
9. Kiểm tra cấu trúc JSON thực tế trước khi xác định schema dữ liệu chính thức.

Các trường như chức danh, công ty, địa điểm, mô tả, loại hợp đồng, `salary_min` và `salary_max` chỉ được sử dụng nếu thực sự có trong phản hồi API. Tên trường và mức độ đầy đủ cần được xác nhận từ dữ liệu thực tế.

**Kết quả đầu ra:** Dữ liệu tin tuyển dụng thô cùng thông tin về quá trình thu thập, sẵn sàng cho bước kiểm tra và phân tích.

### Giai đoạn 2 — Thiết lập SQL Server

**Mục tiêu:** Xây dựng nơi lưu trữ và truy vấn dữ liệu thu thập được.

Các công việc:

1. Tạo database cho dự án.
2. Thiết kế bảng theo schema thực tế của dữ liệu Adzuna API.
3. Xác định khóa phù hợp để nhận diện tin tuyển dụng và hạn chế nạp trùng khi thu thập lại.
4. Nạp dữ liệu vào SQL Server bằng Python.
5. Kiểm tra số lượng bản ghi, kiểu dữ liệu, giá trị thiếu và các vấn đề chất lượng dữ liệu.
6. Viết truy vấn thống kê cơ bản để kiểm tra dữ liệu.
7. Kết nối SQL Server với Python bằng SQLAlchemy và pyodbc.

Có thể thiết kế các bảng riêng cho dữ liệu tin tuyển dụng và thông tin về các lần thu thập nếu điều đó phù hợp với quy mô dự án. Cấu trúc bảng cuối cùng cần dựa trên schema thực tế, không nên giả định trước những trường API không cung cấp.

**Kết quả đầu ra:** Database hoạt động, dữ liệu có thể truy vấn và có quy trình nạp dữ liệu có thể lặp lại.

### Giai đoạn 3 — Data Understanding và EDA

**Mục tiêu:** Hiểu cấu trúc, độ đầy đủ và đặc điểm của dữ liệu trước khi xây dựng mô hình.

Thực hiện trong `01_data_understanding.ipynb` và `02_eda.ipynb`.

Các công việc:

- Xác định số lượng bản ghi, số lượng đặc trưng và kiểu dữ liệu.
- Đọc ý nghĩa các trường được API trả về.
- Phân biệt biến số, biến phân loại, dữ liệu văn bản, trường định danh và biến mục tiêu.
- Kiểm tra dữ liệu thiếu và bản ghi trùng lặp.
- Phân tích thống kê mô tả của các trường số.
- Phân tích tần suất chức danh công việc, địa điểm, loại hợp đồng hoặc các trường phân loại khác nếu có.
- Kiểm tra phân phối của `salary_min`, `salary_max` và `salary` khi có đủ dữ liệu.
- Phát hiện mức lương âm, bằng 0 không hợp lý, sai thứ tự giữa mức tối thiểu và tối đa, hoặc giá trị bất thường.
- Kiểm tra đơn vị tiền tệ, chu kỳ trả lương và tính nhất quán giữa các bản ghi khi thông tin này có sẵn.
- Trực quan hóa phân phối mức lương và mối quan hệ giữa các đặc trưng với biến mục tiêu.
- Đánh giá độ phủ của dữ liệu lương so với toàn bộ tin tuyển dụng.

**Lưu ý:** Không nên trộn lẫn lương theo giờ, theo ngày, theo tháng và theo năm nếu chưa có quy tắc quy đổi đáng tin cậy. Không tự suy ra đơn vị tiền tệ hoặc chu kỳ trả lương khi dữ liệu không xác nhận điều đó.

**Kết quả đầu ra:** Báo cáo EDA, biểu đồ, thống kê mức độ thiếu dữ liệu và danh sách các vấn đề cần giải quyết.

### Giai đoạn 4 — Tiền xử lý dữ liệu

**Mục tiêu:** Chuẩn bị dữ liệu đầu vào phù hợp với các thuật toán Regression.

Thực hiện trong `03_preprocessing.ipynb`, sau đó triển khai thành `src/preprocessing.py`.

Các công việc có thể bao gồm:

- Loại bỏ hoặc xử lý bản ghi trùng lặp dựa trên quy tắc được xác định.
- Xử lý dữ liệu thiếu theo từng loại trường và mục tiêu sử dụng.
- Kiểm tra dữ liệu sai kiểu, giá trị không hợp lệ và các trường có ý nghĩa không rõ ràng.
- Xử lý các bản ghi có `salary_min` hoặc `salary_max` bị thiếu.
- Tạo biến mục tiêu `salary` khi cả hai giá trị mức lương hợp lệ và có thể so sánh trực tiếp.
- Kiểm tra điều kiện `salary_min <= salary_max`.
- Loại bỏ các bản ghi không có nhãn mục tiêu hợp lệ khỏi tập huấn luyện có giám sát, nếu không có phương pháp xử lý được xác định rõ.
- Mã hóa các biến phân loại bằng phương pháp phù hợp, chẳng hạn One-Hot Encoding.
- Xử lý dữ liệu văn bản như chức danh và mô tả công việc nếu được đưa vào mô hình.
- Chuẩn hóa hoặc biến đổi đặc trưng số khi thuật toán yêu cầu.
- Xem xét phân phối lệch của biến lương; chỉ áp dụng biến đổi mục tiêu khi có lý do và có cách diễn giải kết quả rõ ràng.
- Chia dữ liệu thành tập huấn luyện, validation (nếu cần) và kiểm tra.

**Quy tắc tạo biến mục tiêu:**

```python
df["salary"] = (df["salary_min"] + df["salary_max"]) / 2
```

Chỉ áp dụng công thức trên cho các bản ghi có cả `salary_min` và `salary_max` hợp lệ, cùng đơn vị tiền tệ và chu kỳ trả lương có thể so sánh. Trong mã thực tế, cần kiểm tra giá trị thiếu, kiểu dữ liệu và tính hợp lệ trước khi tính.

**Nguyên tắc quan trọng:** Tránh rò rỉ dữ liệu (*data leakage*). Các bước học tham số từ dữ liệu, chẳng hạn điền giá trị thiếu, mã hóa hoặc chuẩn hóa, phải được `fit` trên tập huấn luyện rồi mới áp dụng cho validation và test. Không đưa các trường được dùng trực tiếp để tạo nhãn mục tiêu vào đầu vào mô hình theo cách làm lộ đáp án.

**Kết quả đầu ra:** Quy trình tiền xử lý nhất quán, có thể tái sử dụng khi huấn luyện và dự đoán.

### Giai đoạn 5 — Feature Engineering

**Mục tiêu:** Xây dựng biểu diễn đầu vào có ích cho việc dự đoán mức lương.

Thực hiện trong `04_feature_engineering.ipynb` và triển khai các thành phần ổn định trong `src/feature_engineering.py`.

Các hướng khảo sát, tùy theo trường dữ liệu thực tế:

- Chuẩn hóa và gom nhóm chức danh công việc khi có cơ sở rõ ràng.
- Khai thác thông tin địa điểm, khu vực hoặc quốc gia nếu có.
- Sử dụng loại hợp đồng, hình thức làm việc hoặc cấp bậc nếu API cung cấp.
- Trích xuất đặc trưng từ mô tả công việc, chẳng hạn TF-IDF, nếu nội dung và phạm vi dự án phù hợp.
- Đánh giá việc sử dụng tên công ty hoặc các trường có độ phân biệt cao; tránh để mô hình chỉ ghi nhớ các giá trị xuất hiện trong tập huấn luyện.
- Loại bỏ trường định danh, URL và thông tin không phù hợp làm đặc trưng dự đoán.
- So sánh các nhóm đặc trưng để xác định nhóm nào thực sự cải thiện hiệu quả trên validation.

Không phải tất cả các hướng trên đều bắt buộc. Chỉ sử dụng các đặc trưng có sẵn, có ý nghĩa và có thể được cung cấp khi dự đoán một tin tuyển dụng mới.

**Kết quả đầu ra:** Danh sách đặc trưng được lựa chọn, cách biến đổi tương ứng và bằng chứng thực nghiệm về tác dụng của chúng.

### Giai đoạn 6 — Huấn luyện và thử nghiệm mô hình

**Mục tiêu:** So sánh các thuật toán hồi quy và tìm mô hình phù hợp với dữ liệu.

Thực hiện trong `05_model_experiments.ipynb`.

Các thuật toán dự kiến khảo sát:

- Dummy Regressor — baseline tham chiếu.
- Linear Regression.
- Ridge hoặc Lasso Regression.
- Decision Tree Regressor.
- Random Forest Regressor.
- Gradient Boosting Regressor.
- Các mô hình boosting khác nếu phù hợp với dữ liệu và môi trường.

Danh sách trên là các ứng viên ban đầu, không phải yêu cầu phải sử dụng tất cả. Việc lựa chọn cuối cùng dựa trên kết quả thực nghiệm, độ phức tạp, khả năng diễn giải, thời gian huấn luyện và yêu cầu của dự án.

Các công việc:

1. Xác định cách chia dữ liệu và chỉ số đánh giá trước khi thử nghiệm.
2. Xây dựng baseline để có mốc so sánh.
3. Huấn luyện các mô hình trên cùng tập dữ liệu huấn luyện và quy trình đánh giá nhất quán.
4. Dùng validation hoặc cross-validation để so sánh và điều chỉnh tham số khi cần.
5. Ghi lại tham số, đặc trưng, thời gian chạy và kết quả của từng thí nghiệm.
6. Chọn mô hình dựa trên nhiều tiêu chí thay vì chỉ nhìn vào một chỉ số.
7. Chỉ sử dụng tập test cuối cùng để đánh giá khách quan sau khi quá trình lựa chọn mô hình đã hoàn tất.

**Kết quả đầu ra:** Bảng so sánh mô hình và ứng viên được lựa chọn. Kết quả thực nghiệm sẽ được bổ sung sau khi chạy, không giả định trước độ chính xác hay hiệu suất.

### Giai đoạn 7 — Đánh giá và phân tích lỗi

**Mục tiêu:** Đánh giá mức sai số dự đoán và hiểu những trường hợp mô hình hoạt động chưa tốt.

Thực hiện trong `06_evaluation.ipynb` và triển khai các hàm cần thiết trong `src/evaluate.py`.

Các chỉ số có thể sử dụng:

- **MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình, dễ diễn giải theo đơn vị của biến lương.
- **RMSE (Root Mean Squared Error):** Phạt nặng hơn các sai số lớn.
- **\(R^2\) (Coefficient of Determination):** Đo mức độ mô hình giải thích biến thiên của mục tiêu so với baseline trung bình trong điều kiện đánh giá tương ứng.
- **Median Absolute Error:** Có thể bổ sung khi muốn xem xét chỉ số ít nhạy cảm hơn với một số sai số cực lớn.

Các công việc:

1. Tính các chỉ số trên tập đánh giá được xác định trước.
2. So sánh với baseline.
3. Phân tích phân phối phần dư và sai số dự đoán.
4. Kiểm tra các trường hợp dự đoán âm hoặc giá trị không hợp lý nếu mô hình tạo ra chúng.
5. Phân tích sai số theo nhóm chức danh, địa điểm hoặc các nhóm dữ liệu khác khi số lượng mẫu đủ lớn.
6. Kiểm tra ảnh hưởng của giá trị ngoại lệ và phân phối lệch.
7. Xem xét feature importance hoặc các phương pháp giải thích phù hợp nếu mô hình hỗ trợ.

MAE và RMSE có cùng đơn vị với biến mục tiêu. Vì vậy, chỉ diễn giải chúng sau khi đã xác nhận đơn vị tiền tệ và chu kỳ trả lương của dữ liệu. \(R^2\) có thể âm nếu mô hình hoạt động kém hơn baseline tương ứng.

**Kết quả đầu ra:** Báo cáo đánh giá, bảng so sánh, phân tích lỗi và lý do lựa chọn mô hình.

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