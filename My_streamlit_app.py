from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent

model = joblib.load(
    APP_DIR / "breast_cancer_model.joblib"
)

scaler = joblib.load(
    APP_DIR / "breast_cancer_scaler.joblib"
)

label_encoder = joblib.load(
    APP_DIR / "breast_cancer_label_encoder.joblib"
)

st.set_page_config(
    page_title="Breast Cancer Prediction",
)

st.title("Breast Cancer Prediction")
st.write("Enter the measurements and click Predict.")

FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]
data = pd.read_csv(
    APP_DIR / "breast-cancer.csv"
 )

# radius_mean = st.number_input("radius_mean")
# texture_mean = st.number_input("texture_mean")
# perimeter_mean = st.number_input("perimeter_mean")


default_values = data[FEATURES].iloc[0].to_dict()
with st.form("prediction_form"):
    values = {}

    for feature in FEATURES:
        values[feature] = st.number_input(
            feature,
            value=float(default_values[feature]),
            format="%.5f"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    scaled_input = scaler.transform(input_data)

    prediction_code = model.predict(
        scaled_input
    )[0]

    predicted_label = label_encoder.inverse_transform(
        [prediction_code]
    )[0]
    if predicted_label == "B":
        st.success("Predicted diagnosis: Benign")
    else:
        st.warning("Predicted diagnosis: Malignant")
