# Cancer Risk Prediction

Ứng dụng **Streamlit** dự đoán mức độ rủi ro ung thư (`High` / `Medium` / `Low`) từ
17 yếu tố nguy cơ, sử dụng model **XGBoost** đã huấn luyện sẵn.

## Cấu trúc thư mục

```
Cancer-Risk-Prediction/
├── app.py                      # Ứng dụng Streamlit
├── xgb_complete_pipeline.pkl   # Model + label encoder (đã huấn luyện)
├── cancer-risk-factors.csv     # Dữ liệu gốc
├── Cancer_EDA.ipynb            # Notebook phân tích dữ liệu
├── Cancer_ML (1).ipynb         # Notebook huấn luyện model
├── requirements.txt
├── .env / .env.example         # Biến môi trường (MODEL_PATH, STREAMLIT_PORT)
├── .gitignore
└── README.md
```

## Cách chạy

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy ứng dụng
streamlit run app.py
```

Mặc định ứng dụng mở tại `http://localhost:8501` (cấu hình qua `STREAMLIT_PORT`).

## Cấu hình

Sao chép `.env.example` thành `.env` và chỉnh nếu cần:

```
MODEL_PATH=xgb_complete_pipeline.pkl
STREAMLIT_PORT=8501
```

## Lưu ý kỹ thuật

- **Scaler chưa được fit:** file `.pkl` chứa `StandardScaler` nhưng **chưa fit**, vì vậy
  ứng dụng **không sử dụng** scaler này. Model nhận trực tiếp DataFrame 17 cột thô đúng
  thứ tự.
- Thứ tự 17 features: `Age, Gender, Smoking, Alcohol_Use, Obesity, Family_History,
  Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake, Physical_Activity,
  Air_Pollution, Occupational_Hazards, BRCA_Mutation, H_Pylori_Infection,
  Calcium_Intake, BMI, Physical_Activity_Level`.
- Hiệu năng tham khảo: macro-F1 ≈ 0.73, accuracy ≈ 0.88. Lớp `High` là lớp hiếm (~5.1%)
  nên dễ bị bỏ sót.

## ⚠️ Disclaimer y tế

Ứng dụng chỉ dùng cho mục đích học tập và tham khảo. Kết quả **không thay thế chẩn đoán
hoặc tư vấn y khoa chuyên nghiệp**. Vui lòng liên hệ cơ sở y tế để được đánh giá chính xác.
