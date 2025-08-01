# 🧠 Multi-Disease Prediction Web App

A Streamlit-powered interactive dashboard that predicts the likelihood of multiple diseases (Kidney, Liver, Parkinson’s) based on medical parameters. It uses trained machine learning models and visual explanations (SHAP) to support early detection and proactive healthcare.

---

## 📌 Project Overview

This project enables:

- ✅ **Prediction of multiple diseases** (Kidney, Liver, Parkinson’s)
- 🧠 **SHAP Explainability** for individual predictions
- 📊 **Exploratory Data Analysis (EDA)** with Plotly
- 📂 **Bulk prediction via CSV upload**
- 🧾 **Clean UI** using Streamlit with tabbed navigation
- 🔐 Local-first design (no data leaves your system)

---

## 🧱 System Architecture

### 🔸 Frontend
- Built with **Streamlit** (Python-based web interface)
- Allows users to enter test results or upload CSVs
- Displays predictions, SHAP charts, and performance

### 🔸 Backend
- Processes user input
- Performs data preprocessing, reshaping
- Loads ML models from `.pkl` files and returns predictions

---

## 🧠 ML Models Used

| Disease      | Algorithm            |
|--------------|----------------------|
| Kidney       | XGBoost Classifier   |
| Liver        | Random Forest Classifier |
| Parkinson’s  | Logistic Regression  |

Each model is trained on cleaned data and saved as a `.pkl` file using `joblib` or `pickle`.

---

## 🛠 Features

- 🔬 **Single Record Prediction** with form inputs
- 📊 **EDA**: Histogram, heatmap, feature exploration
- 📁 **CSV Upload**: Bulk predictions with file validation
- 🔎 **SHAP Explainability** for each disease
- 🧩 **Error Handling**: Proper feedback for wrong uploads

---

## 📁 Folder Structure

```
📦multi_disease_predictor/
├── README.md                            # Full project overview
├── requirements.txt                     # List of dependencies
├── app_main.py                          # Streamlit dashboard
├── templates/                           # For CSV upload samples
│   ├── kidney_template.csv
│   ├── liver_template.csv
│   └── parkinsons_template.csv
├── models/                              # Trained ML models
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   └── parkinsons_model.pkl
├── data/                                # Cleaned data for EDA
│   ├── kidney_cleaned.csv
│   ├── liver_cleaned.csv
│   └── parkinsons_cleaned.csv
├── notebooks/                           # Jupyter notebooks
│   ├── 01_kidney_preprocessing_eda.ipynb
│   ├── 02_kidney_model_training.ipynb
│   ├── 03_liver_preprocessing_eda.ipynb
│   ├── 04_liver_model_training.ipynb
│   ├── 05_parkinsons_preprocessing_eda.ipynb
│   └── 06_parkinsons_model_training.ipynb

```

---

## 🚀 How to Run the App

### 1️⃣ Set up virtual environment
```bash
python -m venv .venv
.venv\\Scripts\\activate      # Windows
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, here are key packages:
```bash
pip install streamlit pandas numpy shap matplotlib plotly xgboost scikit-learn
```

### 3️⃣ Launch the app
```bash
streamlit run app_main.py
```

---

## 📂 CSV Upload Instructions

- Go to **Bulk Prediction** tab
- Upload a CSV file with the same structure as used during training
- Click **Predict CSV**
- Download results as a new CSV

⚠️ If format mismatches, the app will display an error like:
> ❌ File format mismatch. Please upload a CSV file with the correct column structure for the selected disease model.

You can also **download CSV templates** for each disease.

---

## 🧪 Feature Input Examples

Each disease has its own form with relevant features.

- **Kidney Disease**: 14 numerical + 11 categorical features = 25 total
- **Liver Disease**: 9 numerical + 1 gender field = 10 total
- **Parkinson’s**: 22 continuous features (e.g. jitter, shimmer, RPDE)

---

## 📈 SHAP Explainability

- Every prediction comes with SHAP waterfall plots
- Users can interpret which features pushed the prediction towards a disease

---

**Tools Used**: Python, Streamlit, Scikit-learn, XGBoost, SHAP, Plotly

---

## 📌 Future Improvements

- Add login/auth system for users
- Save patient prediction history
- Integrate with health monitoring APIs
- Export visual PDF reports

---
## ✅ Status: Complete

All features tested, validated, and documented. The system supports robust, multi-disease prediction workflows.
