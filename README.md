# Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines
Predicts the likelihood that a survey respondent received the H1N1 and seasonal flu vaccines, using background, opinion, and behavioral features from the United States Centers for Disease Control's 2009 National H1N1 Flu Survey.

[![Live Demo](https:/img.shields.io/badge/Live%20Demo-Streamlit-red)](https://url.streamlit.app)
[![Competition](https:/img.shields.io/badge/DrivenData-%2366-blue)](https://www.drivendata.org/competitions/66/flu-shot-learning/)

---
## Problem
Public health agencies need to understand what drives vaccine acceptance and hesitancy to design effective outreach campaigns. This project builds a multi-label classifier that predicts two independent binary outcomes per respondent: HINI vaccine uptake and seasonal flu vaccine uptake.

**Data:** 26,707 survey respondents | 35 features | U.S. CDC NHFS 2009

**Metric:** ROC AUC averaged across both labels

**Type:** Multi-label binary classification

---
## Results

| Model | H1N1 AUC | Seasonal AUC | Mean AUC |
|-------|----------|--------------|----------|
| Logistic Regression (baseline) | | | |
| XGBoost (tuned) | | | |
| XGBoost + LightGBM ensemble | | | |

---
## Key Findings

*Summary of key findings forthcoming*

---
## Pipeline
1. **Problem definition** — framed as multi-label classification with ROC AUC metric
2. **Data loading** — merged features and labels on respondent_id
3. **Cleaning** — median imputation for numeric, mode + label encoding for categorical
4. **EDA** — identified doctor recommendation effect and opinion feature signal
5. **Feature engineering** — opinion composites, behavior score, healthcare access index
6. **Training** — two independent XGBoost models (one per label) + LightGBM ensemble
7. **Validation** — 5-fold Stratified KFold; CV mean AUC 0.868
8. **Hyperparameter tuning** — Optuna (50 trials), all runs logged to MLflow
9. **Explainability** — SHAP feature importance for both models
10. **Deployment** — FastAPI REST API containerized with Docker
11. **Monitoring** — KS-test drift detection, configurable retraining triggers
12. **Demo** — interactive Streamlit app hosted on Streamlit Community Cloud

---
## Tech Stack

Python | pandas | scikit-learn | XGBoost | LightGBM | SHAP | Optuna | MLflow | FastAPI | Docker | Streamlit

---
## Project Structure

```
flu-shot-learning/
├── data/
│   ├── raw/             # Raw competition files (not committed)
│   ├── processed/       # Cleaned and engineered datasets
├── notebooks/           # EDA and experimentation
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py      # Feature engineering functions
│   ├── train.py         # Training and MLflow logging script
│   └── api.py           # FastAPI deployment
├── models/              # Saved model files (not committed)
├── submissions/         # Competition CSV files
├── logs/                # Prediction and monitoring logs
├── app.py               # Streamlit demo app
├── Dockerfile
├── requirements.txt
```
---
## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/flu-shot-learning.git
cd flu-shot-learning
python -m venv flu_env && source flu_env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To launch the MLflow experiment dashboard:
```bash
mlflow ui   # opens at http://localhost:5000
```

To run the prediction API:
```bash
uvicorn src.api:app --reload    #opens at http://localhost:8000
```
---
## Competition

DrivenData. (2015). *Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines.* Retrieved May 10, 2026 from https://www.drivendata.org/competitions/66/flu-shot-learning/