# Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines
Predicts the likelihood that a survey respondent received the H1N1 and seasonal flu vaccines, using background, opinion, and behavioral features from the U.S. CDC's 2009 National H1N1 Flu Survey.

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

---

## Pipeline
1. **Problem Definition**
2. **Data loading**
3. **Cleaning**
4. **EDA**
5. **Feature Engineering**
6. **Training**
7. **Validation**
8. **Hyperparameter tuning**
9. **Explainability**
10. **Deployment**
11. **Monitoring**
12. **Demo**

---

## Tech Stack

Python | pandas | scikit-learn | XGBoost | LightGBM | SHAP | Optuna | MLflow | FastAPI | Docker | Streamlit

---

## Project Structure

flu-shot-learning/
├── data/raw/            # Raw competition files (not committed)
├── data/processed/      # Cleaned and engineered datasets
├── notebooks/           # EDA and experimentation
├── src/
│   ├── features.py      # Feature engineering functions
│   ├── train.py         # Training and MLflow logging pipeline
│   └── api.py           # FastAPI prediction endpoint
├── models/              # Saved model files (not committed)
├── submissions/         # Competition submission CSVs
├── app.py               # Streamlit demo app
├──

---

DrivenData. (2015). *Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines.* Retrieved May 10, 2026 from https://www.drivendata.org/competitions/66/flu-shot-learning/