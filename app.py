"""Ứng dụng Streamlit dự đoán mức độ rủi ro ung thư.

Chạy: streamlit run app.py
"""

import os

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "xgb_complete_pipeline.pkl")

# Đúng thứ tự 17 features mà model đã được huấn luyện.
FEATURES = [
    "Age",
    "Gender",
    "Smoking",
    "Alcohol_Use",
    "Obesity",
    "Family_History",
    "Diet_Red_Meat",
    "Diet_Salted_Processed",
    "Fruit_Veg_Intake",
    "Physical_Activity",
    "Air_Pollution",
    "Occupational_Hazards",
    "BRCA_Mutation",
    "H_Pylori_Infection",
    "Calcium_Intake",
    "BMI",
    "Physical_Activity_Level",
]

st.set_page_config(page_title="Dự đoán rủi ro ung thư", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load pipeline (dict) qua joblib. Không dùng scaler vì chưa được fit."""
    return joblib.load(path)


def main():
    st.title("🩺 Dự đoán mức độ rủi ro ung thư")
    st.caption(
        "Nhập các yếu tố nguy cơ ở thanh bên trái, sau đó bấm **Dự đoán** để xem kết quả."
    )

    # ---------------- Sidebar: nhập liệu ----------------
    st.sidebar.header("Thông tin bệnh nhân")

    age = st.sidebar.slider("Tuổi (Age)", 25, 90, 55, step=1)

    gender_label = st.sidebar.selectbox("Giới tính (Gender)", ["Nam", "Nữ"], index=0)
    gender = 1 if gender_label == "Nam" else 0

    st.sidebar.subheader("Yếu tố lối sống / môi trường (0–10)")
    smoking = st.sidebar.slider("Hút thuốc (Smoking)", 0, 10, 3)
    alcohol = st.sidebar.slider("Sử dụng rượu bia (Alcohol_Use)", 0, 10, 3)
    obesity = st.sidebar.slider("Béo phì (Obesity)", 0, 10, 4)
    red_meat = st.sidebar.slider("Ăn thịt đỏ (Diet_Red_Meat)", 0, 10, 4)
    salted = st.sidebar.slider(
        "Ăn đồ muối/chế biến sẵn (Diet_Salted_Processed)", 0, 10, 4
    )
    fruit_veg = st.sidebar.slider(
        "Ăn rau quả (Fruit_Veg_Intake)", 0, 10, 5
    )
    physical_activity = st.sidebar.slider(
        "Vận động thể chất (Physical_Activity)", 0, 10, 5
    )
    air_pollution = st.sidebar.slider(
        "Ô nhiễm không khí (Air_Pollution)", 0, 10, 5
    )
    occupational = st.sidebar.slider(
        "Nguy cơ nghề nghiệp (Occupational_Hazards)", 0, 10, 4
    )
    calcium = st.sidebar.slider("Canxi (Calcium_Intake)", 0, 10, 4)
    activity_level = st.sidebar.slider(
        "Mức độ vận động (Physical_Activity_Level)", 0, 10, 5
    )

    st.sidebar.subheader("Tiền sử / yếu tố di truyền")
    family_history = (
        1
        if st.sidebar.selectbox("Tiền sử gia đình (Family_History)", ["Không", "Có"], index=0)
        == "Có"
        else 0
    )
    brca = (
        1
        if st.sidebar.selectbox("Đột biến BRCA (BRCA_Mutation)", ["Không", "Có"], index=0)
        == "Có"
        else 0
    )
    h_pylori = (
        1
        if st.sidebar.selectbox(
            "Nhiễm H. Pylori (H_Pylori_Infection)", ["Không", "Có"], index=0
        )
        == "Có"
        else 0
    )

    bmi = st.sidebar.number_input(
        "Chỉ số BMI", min_value=15.0, max_value=41.4, value=26.5, step=0.1, format="%.1f"
    )

    predict_clicked = st.sidebar.button("Dự đoán", type="primary")

    # ---------------- Nội dung chính ----------------
    if predict_clicked:
        if not os.path.exists(MODEL_PATH):
            st.error(
                f"Không tìm thấy file model tại `{MODEL_PATH}`.\n\n"
                "Hãy kiểm tra biến `MODEL_PATH` trong file `.env`, hoặc đảm bảo file "
                "`xgb_complete_pipeline.pkl` nằm cùng thư mục với `app.py`."
            )
            return

        try:
            pipeline = load_model(MODEL_PATH)
            model = pipeline["model"]
            label_encoder = pipeline["label_encoder"]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Không thể tải model: {exc}")
            return

        row = {
            "Age": age,
            "Gender": gender,
            "Smoking": smoking,
            "Alcohol_Use": alcohol,
            "Obesity": obesity,
            "Family_History": family_history,
            "Diet_Red_Meat": red_meat,
            "Diet_Salted_Processed": salted,
            "Fruit_Veg_Intake": fruit_veg,
            "Physical_Activity": physical_activity,
            "Air_Pollution": air_pollution,
            "Occupational_Hazards": occupational,
            "BRCA_Mutation": brca,
            "H_Pylori_Infection": h_pylori,
            "Calcium_Intake": calcium,
            "BMI": float(bmi),
            "Physical_Activity_Level": activity_level,
        }
        X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

        try:
            pred = model.predict(X)
            pred_label = label_encoder.inverse_transform(pred)[0]
            proba = model.predict_proba(X)[0]
            classes = list(label_encoder.classes_)  # ['High', 'Low', 'Medium']
        except Exception as exc:  # noqa: BLE001
            st.error(f"Lỗi khi dự đoán: {exc}")
            return

        label_vi = {"High": "Cao", "Medium": "Trung bình", "Low": "Thấp"}
        color = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
        pred_vi = label_vi.get(pred_label, pred_label)

        st.subheader("Kết quả dự đoán")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Mức độ rủi ro", f"{color.get(pred_label, '')} {pred_vi}")
        with col2:
            proba_df = pd.DataFrame(
                {
                    "Mức độ": [label_vi.get(c, c) for c in classes],
                    "Xác suất": [float(p) for p in proba],
                }
            )
            st.bar_chart(proba_df.set_index("Mức độ"))

        st.write("**Xác suất từng mức độ:**")
        cols = st.columns(len(classes))
        for i, c in enumerate(classes):
            cols[i].metric(label_vi.get(c, c), f"{proba[i] * 100:.1f}%")

        st.warning(
            "⚠️ **Cảnh báo y tế:** Kết quả này chỉ mang tính tham khảo và **không thay thế "
            "chẩn đoán hoặc tư vấn của bác sĩ**. Hãy liên hệ cơ sở y tế để được đánh giá "
            "chính xác."
        )

    else:
        st.info("👈 Nhập thông tin ở thanh bên trái và bấm **Dự đoán**.")

    # ---------------- Thông tin model ----------------
    with st.expander("ℹ️ Thông tin model"):
        st.markdown(
            """
- **Thuật toán:** XGBoost (multi:softmax), 3 lớp: `High`, `Low`, `Medium`.
- **Hiệu năng (theo notebook huấn luyện):** macro-F1 ≈ **0.73**, accuracy ≈ **0.88**.
- **Lưu ý về lớp hiếm:** lớp `High` chỉ chiếm khoảng **5.1%** dữ liệu, nên khả năng
  nhận diện lớp này thấp hơn hai lớp còn lại.
- **Scaler:** pipeline có kèm `StandardScaler` **nhưng chưa được fit**, do đó ứng dụng
  **không sử dụng scaler**. Model nhận trực tiếp DataFrame 17 cột thô đúng thứ tự.
"""
        )


if __name__ == "__main__":
    main()
