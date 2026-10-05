# Cancer Risk Prediction

A **Streamlit** app that predicts cancer risk level (`High` / `Medium` / `Low`) from
17 risk factors, using a pre-trained **XGBoost** model.

## Project structure

```
Cancer-Risk-Prediction/
├── app/
│   └── app.py                  # Streamlit app
├── data/
│   └── cancer-risk-factors.csv # Raw dataset
├── models/
│   └── xgb_complete_pipeline.pkl  # Model + label encoder (pre-trained)
├── notebooks/
│   ├── Cancer_EDA.ipynb        # Data analysis notebook
│   └── Cancer_ML (1).ipynb     # Model training notebook
├── requirements.txt
├── run.bat / run.ps1           # Convenience launcher scripts
├── .env / .env.example         # Environment variables (MODEL_PATH, STREAMLIT_PORT)
├── .gitignore
└── README.md
```

## How to run

```bash
# 1. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/app.py
# ...or use the convenience script: run.bat (Windows CMD) / run.ps1 (PowerShell)
```

The app opens at `http://localhost:8501` by default (configurable via `STREAMLIT_PORT`).

## Configuration

Copy `.env.example` to `.env` and adjust if needed:

```
MODEL_PATH=models/xgb_complete_pipeline.pkl
STREAMLIT_PORT=8501
```

## Technical notes

- **Scaler is not fitted:** the `.pkl` file contains a `StandardScaler` that is **not
  fitted**, so the app **does not use** it. The model receives the raw 17-column DataFrame
  directly in the correct order.
- Feature order (17): `Age, Gender, Smoking, Alcohol_Use, Obesity, Family_History,
  Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake, Physical_Activity,
  Air_Pollution, Occupational_Hazards, BRCA_Mutation, H_Pylori_Infection,
  Calcium_Intake, BMI, Physical_Activity_Level`.
- Reference performance: macro-F1 ≈ 0.73, accuracy ≈ 0.88. `High` is a rare class (~5.1%)
  and is therefore easy to miss.

## ⚠️ Medical disclaimer

This app is for educational and reference purposes only. The results **do not replace
professional diagnosis or medical advice**. Please consult a healthcare provider for an
accurate assessment.
