# Telecom-Customer-Churn-Predictor
An interactive, production-ready Streamlit web application that uses a trained XGBoost classifier to predict customer churn risks based on demographic, service, and account information.
---

## 🚀 Live Demo
🔗 **[Access the Deployed App Here](https://telecom-predictor.streamlit.app/)**

---

## 🛠️ Project Structure

```text
├── Data_Cleaning_and _Exploration.ipynb
├── Model_Building.ipynb
├── Model_Explainability.ipynb
├── app.py                # Streamlit UI and custom feature engineering
├── XGB_model_2.pkl       # Serialized Scikit-Learn / XGBoost Pipeline
├── requirements.txt      # Production dependencies for Streamlit Cloud
└── README.md             # Project documentation
```

## 📈 End-to-End Prediction Pipeline
Rather than relying on manual, brittle preprocessing inside the web app code, this project leverages a unified Scikit-Learn Pipeline framework. This ensures that the training environment and production web application handle data identically.

### 1. Custom Feature Engineering (wrangle_input)
Before hitting the machine learning pipeline, raw user inputs from the UI are structured and transformed:

* Temporal Conversion: Translates raw customer tenure from months to years.

* Loyalty Tiers: Derived rule-based metrics to segment users into New, Mid, and Loyal categories.

* Financial Metrics: Computes a calculated Payment Frequency based on the ratio of total charges to monthly charges.

* Column Realignment: Automatically drops administrative columns (like customerID) to preserve the expected feature space dimension.

### 2. Embedded Model Pipeline (model_2)
The exported XGB_model_2.pkl contains a unified execution chain. When model.predict() is called, the web app pipes data seamlessly through:

* preprocessor (Automatic Ordinal Encoding of raw text features)

* Scaling (Robust data normalization via StandardScaler)

* Xgb Boost (High-performance gradient-boosted classification trees)
  
#### 📊 Model Performance Evaluation (Minority Class)
Because customer churn is highly class-imbalanced, the optimization targets minority class (`Class 1`) viability over generic baseline accuracy. The deployed model configuration achieves:
* **Recall (Minority Class):** `0.76` — Successfully captures 76% of all customers who will ultimately leave.
* **F1-Score (Minority Class):** `0.64` — Achieves a robust harmonic balance between precision and detection rates.

### Note: Synthetic Minority Over-sampling (SMOTE) is safely embedded within the file structure and automatically bypassed during evaluation/inference.

## 🧰 Tech Stack
* Frontend UI Framework: Streamlit

* Data Core: Pandas, NumPy

* Machine Learning Pipeline: Scikit-Learn

* Sampling Architecture: Imbalanced-Learn (SMOTE)

* Inference Engine: XGBoost
