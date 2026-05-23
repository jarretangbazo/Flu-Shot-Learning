<!-- omit in toc -->
# Flu Shot Learning
<!-- omit in toc -->
### Predict H1N1 and Seasonal Flu Vaccines
**DrivenData Competition:** https://www.drivendata.org/competitions/66/flu-shot-learning/

Predict the likelihood that a survey respondent received the H1N1 and seasonal flu vaccines, using background, opinion, and behavioral features from the United States Centers for Disease Control's 2009 National H1N1 Flu Survey.

[![Live Demo](https:/img.shields.io/badge/Live%20Demo-Streamlit-red)](https://url.streamlit.app)
[![Competition](https:/img.shields.io/badge/DrivenData-%2366-blue)](https://www.drivendata.org/competitions/66/flu-shot-learning/)

---
<!-- omit in toc -->
## Table of Contents

- [Competition](#competition)
  - [Problem Definition](#problem-definition)
  - [Task](#task)
  - [Getting the Data](#getting-the-data)
    - [Data Files](#data-files)
- [Environment Setup](#environment-setup)
  - [Project Structure](#project-structure)
  - [Local Setup](#local-setup)
  - [Tech Stack](#tech-stack)
- [Results and Key Findings](#results-and-key-findings)
  - [Results](#results)
  - [Key Findings](#key-findings)


---

## Competition

### Problem Definition
Public health agencies need to understand what drives vaccine acceptance and hesitancy to design effective outreach campaigns. This project builds a multi-label classifier that predicts two independent binary outcomes per respondent: HINI vaccine uptake and seasonal flu vaccine uptake.

**Why it matters**
Understanding what drives vaccine uptake (and hesitancy) allows public health agencies to make better decisions about where to direct campaign resources. A campaign that reaches everyone equally wastes budget on people who would have vaccinated anyway, while a more targeted approach concentrates outreach on the populations least likely to vaccinate and most likely to respond. Increases in vaccination resulting from better targeting translates directly into avoided hospitalizations and reduced strain on healthcare systems. This applies directly to future pandemics, including COVID-19.

### Task

Given survey responses about a person's background, opinions, and behaviors, we will predict two binary outcomes:
- Did they receive the **H1N1 vaccine**? (`h1n1_vaccine`: 0 or 1)
- Did they receive the **seasonal flu vaccine**? (`seasonal_vaccine`: 0 or 1)

### Getting the Data

1. Sign up or log in at https://www.drivendata.org
2. Join the competition at the link above
3. Go to the **Data** tab and download all files into `data/raw/`

#### Data Files

| File | Description |
|------|-------------|
| `training_set_features.csv` | Survey responses for ~26,707 training respondents |
| `training_set_labels.csv` | The two vaccine targets for each training respondent |
| `test_set_features.csv` | Survey responses for ~26,708 test respondents |
| `submission_format.csv` | Template showing the required submission structure |

## Environment Setup

### Project Structure

```
flu-shot-learning/
├── data/
│   ├── raw/                                    # Raw competition files (not committed)
│   ├── processed/                              # Cleaned and engineered datasets
├── notebooks/                                  # EDA and experimentation
│   ├── 01_eda.ipynb                            # Exploratory data analysis
│   ├── 02_data_cleaning.ipynb                  # Imputation, encoding, missingness indicators
│   └── 03_feature_engineering.ipynb            # Engineered features
│   └── 04_baseline_models.ipynb                # Logistic regression, decision tree, random forest
│   └── 05_advanced_models.ipynb                # LightGBM tuning, XGBoost, ensemble, submission
│   └── 06_model_evaluation.ipynb               # SHAP, permutation importance, subgroup AUC
├── reports/           
│   ├── figures                                 # All saved plots
├── models/                                     # Saved model files (not committed)
├── submissions/                                # Competition CSV files
├── requirements.txt
```

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/flu-shot-learning.git
cd flu-shot-learning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Run notebooks in order: `01_eda` -> `02_data_cleaning` -> `03_feature_engineering` -> `04_baseline_models` -> `05_advanced_models` -> `06_model_evaluation`

### Tech Stack

Python | pandas | scikit-learn | matplotlib | seaborn | LightGBM | XGBoost

---

## Results and Key Findings

### Results

Two binary classifiers were trained and evaluated using ROC-AUC averaged across both targets. All scores are out-of-fold estimates from stratified 5-fold cross-validation, stratified on `h1n1_vaccine` (the minority class at 21.2% positive).

| Model | H1N1 AUC | Seasonal AUC | Mean AUC |
|-------|----------|--------------|----------|
| Majority-class baseline | 0.500 | 0.500 | 0.500 |
| Logistic Regression | 0.821 | 0.845 | 0.833 |
| Random Forest (n=300) | 0.827 | 0.852 | 0.839 |
| LightGBM (default) | | | |
| LightGBM (tuned) | | | |
| XGBoost | | | |
| Ensemble (weighted avg) | | | |

### Key Findings

**Data Quality**

- 30 of 35 features had at least some missing values. Three features had severe structural missingness: `employment_occupation` (50.4%), `employment_industry` (49.9%), and `health_insurance` (45.9%).
- `doctor_recc_h1n1` and `doctor_recc_seasonal` are always missing together. We create a single missingness indicator for these features in preprocessing.
- Opinion column missingness is **MNAR**: respondents who skipped vaccine perception questions had 7.6-10.3 percentage points lower seasonal vaccination rates. Their non-response is itself a predictive signal, captured by a `missing_opinion_count` engineered feature.

**What predicts vaccination**

Features ranked by predictive strength:

| **Tier** | **Features** | **Signal** |
|----------|--------------|------------|
| 1. Physician | `doctor_recc_h1n1` (r=0.394), `doctor_recc_seasonal` (r=0.369) | Strongest predictors for their respective targets |
| 2. Attitudes | `opinion_*_risk`, `opinion_*_vacc_effective` (r=0.27-0.39) | Skepticism about efficacy and personal risk |
| 3. Demographics | `age_group`, `income_poverty`, `race` | Modest signal; important for equity analysis |
| - | Behavioral features (handwashing, avoidance, etc.) | Near-zero correlation with vaccination uptake |

Cross-target signal is meaningful: H1N1 opinion features predict seasonal vaccination at r~0.18-0.22, and vice versa. All features were included in both models.

**Feature Engineering**

Six features were engineered from EDA findings and validated against held-out AUC before inclusion:

| **Feature** | **Basis** |
|-------------|-----------|
| `missing_opinion_count` | MNAR non-response signal. 10pp lower seasonal rate when missing |
| `high_vacc_ind_occ` | Two obfuscated codes with highest vaccination rates in dataset (62% H1N1, 84% seasonal) |
| `doctor_recc_both` | Interaction of the two strongest individual predictors |
| `opinion_h1n1_composite` | Mean of three H1N1 opinion features (r=0.27-0.32 individually) |
| `opinion_seas_composite` | Mean of three seasonal opinion features (r=0.36-0.39 individually) |
| `behavior_composite` | Mean of seven behavioral features (each r<0.12 individually) |

`census_msa` was dropped. Less than 2 percentage points of vaccination rate spread across all three categories.

**Class Imbalance**

- `h1n1_vaccine`: 21.2% positive, 3.71:1 imbalance -> `scale_pos_weight=3.71` in LightGBM
- `seasonal_vaccine`: 46.6% positive, 1.15:1 -> near-balanced, standard training

**Recommendation**

The model was used to generate prioritized resource allocation recommendations for public health planning:

1. **Physician engagement programs:** Doctor recommendation is the single strongest predictor for both targets. Resources directed in this direction have the highest expected return of any intervention.
2. **Targeted messaging for the hesitant:** Non-vaccinators are skeptical about vaccine efficacy and personal risk and are not constrained by access. Credible, physician-endorsed messaging about what vaccines do and who is at risk is more likely to move this group than a generic awareness campaign.
3. **Equity review before deployment:** A model that performs well on average can still under-serve populations. Subgroup performance (by age, education, race, income) must be reviewed before any targeting system is operationalized.

**Do not prioritize:** General behavioral promotion campaigns (e.g., handwashing, social distancing). These behaviors show near-zero correlation with vaccination uptake and are not a viable proxy for vaccine receptivity. 

---

DrivenData. (2015). *Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines.* Retrieved May 10, 2026 from https://www.drivendata.org/competitions/66/flu-shot-learning/.