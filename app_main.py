import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

models = {
    "Kidney": load_model("kidney_model.pkl"),
    "Liver": load_model("liver_model.pkl"),
    "Parkinson’s": load_model("parkinsons_model.pkl")
}

# -----------------------------
# Prediction Function
# -----------------------------
def predict(model, input_data):
    data = np.array(input_data).reshape(1, -1)
    prob = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]
    return pred, prob, data

# -----------------------------
# SHAP Explainability
# -----------------------------
def shap_explain(model, data, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    st.subheader("🧠 SHAP Explanation")
    st.markdown("**Feature Impact (positive vs negative):**")

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")

# -----------------------------
# Show Prediction Result
# -----------------------------
def show_result(disease, result, prob):
    status = "🔵 Positive" if result == 1 else "🟢 Negative"
    st.metric(label=f"{disease} Prediction", value=status)
    st.metric("Confidence", f"{prob * 100:.2f}%")
    fig = px.pie(values=[prob, 1 - prob], names=["Disease", "No Disease"],
                 title=f"{disease} Probability", hole=0.5,
                 color_discrete_sequence=["red", "green"])
    st.plotly_chart(fig)

# -----------------------------
# Bulk CSV Prediction
# -----------------------------
def csv_predict(model, uploaded_file, disease):
    df = pd.read_csv(uploaded_file)

    # Drop any target/label columns if present
    for col in ["target", "status", "classification", "dataset"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    try:
        # Try prediction
        pred = model.predict(df)
        prob = model.predict_proba(df)[:, 1]
        df["Prediction"] = pred
        df["Probability"] = prob

        st.success("✅ Prediction Completed")
        st.dataframe(df)

        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Predictions",
            data=csv_download,
            file_name=f"{disease.lower()}_bulk_prediction.csv",
            mime="text/csv"
        )
    except ValueError as e:
        st.error("❌ File format mismatch. Please upload a CSV file with the correct column structure for the selected disease model.")
        st.code(str(e), language="text")



# -----------------------------
# EDA Section
# -----------------------------
def eda_section(dataset_path, target_col):
    df = pd.read_csv(dataset_path)
    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head())

    st.markdown("### 📈 Target Distribution")
    st.plotly_chart(px.histogram(df, x=target_col, color=target_col))

    st.markdown("### 🔥 Correlation Heatmap")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig)

    st.markdown("### 🔍 Feature Distribution (select)")
    selected_col = st.selectbox("Select feature", df.columns)
    st.plotly_chart(px.histogram(df, x=selected_col))

# -----------------------------
# UI Starts Here
# -----------------------------
st.set_page_config("Multiple Disease Prediction", layout="wide")
st.title("🧠 Multi-Disease Prediction App")

tabs = st.tabs(["🏥 Predict Disease", "📊 Exploratory Data Analysis", "📂 Bulk CSV Prediction"])

# -----------------------------
# Tab 1: Disease Prediction
# -----------------------------
with tabs[0]:
    disease = st.selectbox("Select Disease", ["Kidney", "Liver", "Parkinson’s"])

    if disease == "Parkinson’s":
        st.subheader("🎤 Parkinson’s Prediction Form")
        features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "Jitter(%)", "Jitter(Abs)", "RAP", "PPQ", "DDP",
                    "Shimmer", "Shimmer(dB)", "APQ3", "APQ5", "APQ", "DDA", "NHR", "HNR", "RPDE", "DFA", "spread1",
                    "spread2", "D2", "PPE"]
        values = [st.number_input(f, key=f"pd_{i}") for i, f in enumerate(features)]
        if st.button("🔍 Predict Parkinson’s"):
            pred, prob, reshaped = predict(models[disease], values)
            show_result(disease, pred, prob)
            shap_explain(models[disease], reshaped, features)

    elif disease == "Liver":
        st.subheader("🦢 Liver Prediction Form")
        age = st.number_input("Age", key="liver_age")
        total_bilirubin = st.number_input("Total Bilirubin", key="liver_tb")
        direct_bilirubin = st.number_input("Direct Bilirubin", key="liver_db")
        alk_phos = st.number_input("Alkaline Phosphotase", key="liver_alk")
        alt = st.number_input("Alamine Aminotransferase", key="liver_alt")
        ast = st.number_input("Aspartate Aminotransferase", key="liver_ast")
        total_proteins = st.number_input("Total Proteins", key="liver_tp")
        albumin = st.number_input("Albumin", key="liver_albumin")
        agr = st.number_input("Albumin and Globulin Ratio", key="liver_agr")
        gender = st.selectbox("Gender", ["Male", "Female"], key="liver_gender")
        gender_val = 1 if gender == "Male" else 0
        values = [age, total_bilirubin, direct_bilirubin, alk_phos, alt, ast,
                  total_proteins, albumin, agr, gender_val]
        if st.button("🔍 Predict Liver Disease"):
            pred, prob, reshaped = predict(models[disease], values)
            show_result(disease, pred, prob)
            shap_explain(models[disease], reshaped, None)

    elif disease == "Kidney":
        st.subheader("💧 Kidney Disease Prediction Form")
        kidney_numerical_fields = [
            "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
            "Blood Glucose Random", "Blood Urea", "Serum Creatinine",
            "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume",
            "WBC Count", "RBC Count"
        ]
        kidney_numerical = [st.number_input(field, key=f"kidney_num_{i}") for i, field in enumerate(kidney_numerical_fields)]

        categorical_options = {
            "Red Blood Cells": ["normal", "abnormal"],
            "Pus Cell": ["normal", "abnormal"],
            "Pus Cell Clumps": ["notpresent", "present"],
            "Bacteria": ["notpresent", "present"],
            "Hypertension": ["no", "yes"],
            "Diabetes Mellitus": ["no", "yes"],
            "Coronary Artery Disease": ["no", "yes"],
            "Appetite": ["good", "poor"],
            "Pedal Edema": ["no", "yes"],
            "Anemia": ["no", "yes"],
            "Smoker": ["no", "yes"]
        }
        cat_map = {
            "normal": 0, "abnormal": 1,
            "notpresent": 0, "present": 1,
            "no": 0, "yes": 1,
            "good": 1, "poor": 0
        }
        kidney_categorical = []
        for i, (label, options) in enumerate(categorical_options.items()):
            selected = st.selectbox(label, options, key=f"kidney_cat_{i}")
            kidney_categorical.append(cat_map[selected])

        input_data = kidney_numerical + kidney_categorical
        st.write(f"🧪 Input shape: {len(input_data)} features")

        if st.button("🔍 Predict Kidney Disease"):
            if len(input_data) != 25:
                st.error(f"❌ Feature count mismatch: Expected 25, got {len(input_data)}")
            else:
                pred, prob, reshaped = predict(models[disease], input_data)
                show_result(disease, pred, prob)
                shap_explain(models[disease], reshaped, None)

# -----------------------------
# Tab 2: EDA
# -----------------------------
with tabs[1]:
    st.subheader("📈 Exploratory Data Analysis")
    selected = st.selectbox("Choose Dataset", ["Kidney", "Liver", "Parkinson’s"])
    dataset_map = {
        "Kidney": "kidney_cleaned.csv",
        "Liver": "liver_cleaned.csv",
        "Parkinson’s": "parkinsons_cleaned.csv"
    }
    target_map = {
        "Kidney": "classification",
        "Liver": "dataset",
        "Parkinson’s": "status"
    }
    eda_section(dataset_map[selected], target_map[selected])

# -----------------------------
# Tab 3: CSV Upload
# -----------------------------
with tabs[2]:
    st.subheader("📂 Bulk Prediction via CSV")
    selected = st.selectbox("Choose Disease Model", ["Kidney", "Liver", "Parkinson’s"], key="csv_disease")
    file = st.file_uploader("Upload CSV", type="csv")

    if file and st.button("📤 Predict CSV"):
        csv_predict(models[selected], file, selected)
