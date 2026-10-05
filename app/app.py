import os

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_complete_pipeline.pkl")

# Exact order of the 17 features the model was trained on.
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

st.set_page_config(page_title="Cancer Risk Prediction", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load the pipeline (dict) via joblib. Scaler is not used because it is unfitted."""
    return joblib.load(path)


def main():
    st.title("🩺 Cancer Risk Prediction")
    st.caption(
        "Enter the risk factors in the left sidebar, then click **Predict** to see the result."
    )

    # ---------------- Sidebar: inputs ----------------
    st.sidebar.header("Patient information")

    age = st.sidebar.slider("Age", 25, 90, 55, step=1)

    gender_label = (1 if st.sidebar.selectbox("Gender", ["Female", "Male"], index=0) == "Male" else 0)

    st.sidebar.subheader("Lifestyle / environmental factors (0–10)")
    smoking = st.sidebar.slider("Smoking", 0, 10, 3)
    alcohol = st.sidebar.slider("Alcohol Use", 0, 10, 3)
    obesity = st.sidebar.slider("Obesity", 0, 10, 4)
    red_meat = st.sidebar.slider("Diet: Red Meat", 0, 10, 4)
    salted = st.sidebar.slider("Diet: Salted / Processed Food", 0, 10, 4)
    fruit_veg = st.sidebar.slider("Fruit & Vegetable Intake", 0, 10, 5)
    physical_activity = st.sidebar.slider("Physical Activity", 0, 10, 5)
    air_pollution = st.sidebar.slider("Air Pollution", 0, 10, 5)
    occupational = st.sidebar.slider("Occupational Hazards", 0, 10, 4)
    calcium = st.sidebar.slider("Calcium Intake", 0, 10, 4)
    activity_level = st.sidebar.slider("Physical Activity Level", 0, 10, 5)

    st.sidebar.subheader("Medical history / genetic factors")
    family_history = (
        1
        if st.sidebar.selectbox("Family History", ["No", "Yes"], index=0)
        == "Yes"
        else 0
    )
    brca = (
        1
        if st.sidebar.selectbox("BRCA Mutation", ["No", "Yes"], index=0)
        == "Yes"
        else 0
    )
    h_pylori = (
        1
        if st.sidebar.selectbox("H. Pylori Infection", ["No", "Yes"], index=0)
        == "Yes"
        else 0
    )

    bmi = st.sidebar.number_input(
        "BMI", min_value=15.0, max_value=41.4, value=26.5, step=0.1, format="%.1f"
    )

    predict_clicked = st.sidebar.button("Predict", type="primary")

    if predict_clicked:
        if not os.path.exists(MODEL_PATH):
            st.error(
                f"Model file not found at `{MODEL_PATH}`.\n\n"
                "Please check the `MODEL_PATH` variable in your `.env` file, or make sure "
                "`xgb_complete_pipeline.pkl` is located in the `models/` folder."
            )
            return

        try:
            pipeline = load_model(MODEL_PATH)
            model = pipeline["model"]
            label_encoder = pipeline["label_encoder"]
        except Exception as exc:
            st.error(f"Failed to load the model: {exc}")
            return

        row = {
            "Age": age,
            "Gender": gender_label,
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
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            return

        color = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

        st.subheader("Prediction result")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Risk level", f"{color.get(pred_label, '')} {pred_label}")
        with col2:
            proba_df = pd.DataFrame(
                {
                    "Risk level": list(classes),
                    "Probability": [float(p) for p in proba],
                }
            )
            st.bar_chart(proba_df.set_index("Risk level"))

        st.write("**Probability for each risk level:**")
        cols = st.columns(len(classes))
        for i, c in enumerate(classes):
            cols[i].metric(c, f"{proba[i] * 100:.1f}%")

        st.warning(
            "⚠️ **Medical disclaimer:** This result is for reference only and **does not "
            "replace a professional diagnosis or medical advice**. Please consult a "
            "healthcare provider for an accurate assessment."
        )

    else:
        st.info("👈 Enter the information in the left sidebar and click **Predict**.")

    # ---------------- Model information ----------------
    with st.expander("ℹ️ Model information"):
        st.markdown(
            """
- **Algorithm:** XGBoost (multi:softmax), 3 classes: `High`, `Low`, `Medium`.
- **Performance (from the training notebook):** macro-F1 ≈ **0.73**, accuracy ≈ **0.88**.
- **Rare class note:** the `High` class accounts for only about **5.1%** of the data, so
  it is harder to detect than the other two classes.
- **Scaler:** the pipeline includes a `StandardScaler` that is **not fitted**, so the app
  **does not use the scaler**. The model receives the raw 17-column DataFrame directly in
  the correct order.
"""
        )


if __name__ == "__main__":
    main()
