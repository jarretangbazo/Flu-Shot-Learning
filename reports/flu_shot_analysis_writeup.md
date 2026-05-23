# Predicting Flu Vaccination Uptake: From Model to Decision

**Project**: Flu Shot Learning — H1N1 and Seasonal Influenza Vaccination Prediction  
**Author**: Jarret Angbazo  
**Date**: May 2026

---

## The Question Behind the Data

Every autumn, public health agencies, hospital systems, and insurers face the same problem: how do you get more people vaccinated before flu season peaks? The standard answer has been broadcast campaigns — mass mailers, radio spots, pharmacy signs. Everyone gets the same message regardless of whether they were going to get vaccinated anyway.

This is expensive and largely inefficient. Resources spent reaching people who would have acted without prompting are resources that cannot be directed toward people who genuinely need a nudge.

The data underlying this analysis comes from the 2009 National H1N1 Flu Survey (NHFS), collected during a period of genuine public health urgency — the H1N1 pandemic. Respondents were asked about their vaccination status, their attitudes toward vaccines, their health behaviors, and their demographic characteristics. Two outcomes were tracked: whether someone received the seasonal flu vaccine and whether they received the H1N1 vaccine.

The core question this analysis answers is not "can a model predict vaccination status?" It is: **what separates people who get vaccinated from people who don't, and what can we do about it?**

---

## Why This Analysis Matters

Consider two ways a public health organization could approach a vaccination campaign with a fixed budget:

**Option A — Broadcast**: Distribute resources evenly across the target population. Everyone receives the same outreach. The campaign is fair in the sense that no one is excluded, but it is also indiscriminate. A significant share of the budget reaches people who were already planning to get vaccinated.

**Option B — Targeted**: Use a predictive model to identify the subset of the population least likely to get vaccinated but most responsive to outreach. Concentrate resources there. Accept that some people who would not have vaccinated anyway are still missed, but dramatically improve the return on every dollar spent.

The gap between these two strategies is not hypothetical. In a population of 26,707 survey respondents, only 21.2% received the H1N1 vaccine. A well-calibrated model that can identify the high-risk, high-persuadability segment with reasonable precision gives the campaign director a targeting list. The difference between a 21% and a 30% vaccination rate — achievable with effective targeting — translates directly into avoided illnesses, avoided hospitalizations, and reduced strain on the healthcare system.

This is the reason the analysis exists. Not to produce an AUC score. To change what someone does with a campaign budget.

---

## What the Data Tells Us

### The population we are working with

The survey captured 26,707 respondents. The two vaccination outcomes tell different stories:

- **Seasonal flu vaccine**: 46.6% uptake — near even split, modest imbalance
- **H1N1 vaccine**: 21.2% uptake — four out of five respondents did not receive it

These are not small differences. The H1N1 result in particular reflects the genuine hesitancy and confusion that characterized the 2009 pandemic response. It also means any model predicting H1N1 uptake is working against a three-to-one class imbalance — most of the population is a true negative, and identifying the positives requires the model to work harder.

### What actually predicts vaccination

The analysis identified a clear hierarchy of predictive factors. Understanding this hierarchy is the core business output — it tells us not just who is unlikely to vaccinate, but *why*, which determines what kind of intervention has a chance of working.

**Tier 1 — Physician recommendation (strongest signal)**

Doctor recommendation is the single strongest predictor for both targets. Respondents whose physician recommended the H1N1 vaccine were vaccinated at dramatically higher rates than those without a recommendation, with a correlation of 0.394 — the highest of any individual feature. The same holds for seasonal flu.

This finding has an immediate operational implication: **the highest-leverage intervention is upstream of the patient**. If the bottleneck is physician recommendation rates, then the campaign that will move the needle most is one that works through clinical workflows — decision support tools that prompt physicians to make vaccine recommendations during relevant encounters — not one that goes directly to patients with messaging.

**Tier 2 — Attitudes and perceptions (strong signal, addressable)**

Perceived vaccine effectiveness and perceived personal risk are the next strongest predictors. Respondents who believed the H1N1 vaccine was effective and believed they were at risk of getting seriously ill were substantially more likely to be vaccinated. These features correlate with vaccination at 0.27–0.39 depending on the target.

This is important because it identifies the mechanism of hesitancy. People who are not vaccinated are disproportionately likely to be skeptical about whether the vaccine works or whether the disease is serious. That is a persuasion problem, not an access problem. Interventions designed to reduce friction — making vaccines available at more pharmacies, extending hours — will not move this group. What might move them is credible, specific information about vaccine efficacy from a trusted source.

**Tier 3 — Demographics (modest signal, contextual)**

Age is a meaningful predictor for seasonal flu: vaccination rates rise sharply with age, from 28.5% among 18–34 year olds to 67.4% among those 65 and over. This aligns with clinical guidelines that prioritize older adults. For H1N1, the age gradient was more muted — the pandemic-era urgency extended risk perception across age groups.

Income, education, race, and employment status each carry modest predictive signal. These are not strong enough to drive targeting decisions on their own, but they are important for equity analysis: a campaign that concentrates resources effectively but disproportionately bypasses lower-income or minority populations has produced an efficiency gain at a fairness cost that must be weighed explicitly.

**What does not predict vaccination**

The behavioral features — handwashing, avoiding large gatherings, wearing face masks — were weak predictors for both outcomes. Neither were housing tenure or marital status meaningful. This tells us something practically useful: general health consciousness and protective behaviors do not reliably predict vaccination behavior. The factors that matter are specific to the vaccine decision itself — what a doctor said, and what the person believes about vaccine efficacy and personal risk.

### A note on missing data as signal

One of the more nuanced findings of the exploratory analysis: respondents who declined to answer questions about vaccine attitudes had measurably lower vaccination rates — up to 10 percentage points lower for seasonal flu. People who skip questions about whether they think vaccines work are not randomly distributed. Their non-response is itself a signal of disengagement or skepticism. This is the kind of insight that only surfaces through careful data quality work, not through model outputs alone.

---

## The Model

### Approach

Two separate binary classifiers were trained — one for H1N1 uptake, one for seasonal flu uptake. Models were evaluated using ROC-AUC averaged across both targets, which is the competition's official metric and a sensible choice for imbalanced binary classification. A higher AUC means the model more reliably ranks non-vaccinators above vaccinators, which is exactly what a targeting application requires.

LightGBM was selected as the primary model based on its ability to handle the mixed ordinal and binary feature structure of survey data, its native support for class imbalance weighting, and its strong empirical performance on tabular classification tasks. A logistic regression baseline was trained first to establish that the relationship between features and outcomes is partially linear — which it is — and to provide a benchmark the more complex model needed to meaningfully exceed.

### Performance

The final model achieved strong discrimination on both targets, with the seasonal flu model performing better than the H1N1 model — consistent with the greater class imbalance and higher baseline uncertainty of the H1N1 prediction problem. The gap between the logistic regression baseline and the tuned LightGBM model reflects the value of capturing non-linear interactions, particularly between physician recommendation and attitude features.

The model produces probabilities, not binary labels. This is intentional. A targeting application does not need a yes/no decision — it needs a ranked list. The campaign team can choose their own threshold based on their budget and risk tolerance: cover the top 10% of predicted non-vaccinators, or the top 30%, or draw the cutoff wherever expected cost per conversion equals the value of a prevented hospitalization.

### Honest limitations

The model was trained on 2009 pandemic survey data. It reflects the attitudes, demographics, and physician behaviors of that moment. Hesitancy patterns have shifted since then; the landscape of misinformation is different; trust in institutions has changed. A model trained on this data should not be deployed for a 2026 campaign without retraining on contemporary data.

Several features with high predictive power — particularly employment industry and occupation — are obfuscated in the dataset. The analysis treats them as opaque and works only with what the data shows. In a production application, these would be known, and their predictive value could be more fully exploited.

---

## Recommendation

Based on the findings of this analysis, resources should be allocated according to the following priorities:

**Priority 1 — Physician engagement programs**

The data is unambiguous: a doctor's recommendation is the most powerful predictor of vaccination in this population. Campaigns that work through clinical systems — flagging eligible patients in EHRs, prompting physicians during annual visits, building vaccine recommendation into standing orders for high-risk patient groups — have the highest expected return. This is not a marketing problem. It is a clinical workflow problem, and it should be funded and managed accordingly.

**Priority 2 — Targeted messaging for attitude-based hesitancy**

The second tier of non-vaccinators is characterized by skepticism about efficacy and personal risk. These respondents are reachable, but the content of outreach matters. Generic awareness campaigns will not move them. Credible, specific, and physician-endorsed communication about what the vaccine does and who is at risk has a reasonable chance of shifting behavior for this group. The model can identify who they are; the campaign design team determines what to say.

**Priority 3 — Equity review before deployment**

Before any targeting system is operationalized, the subgroup performance analysis must be reviewed. A model that performs well on average can perform poorly for specific demographic groups, producing a targeting system that systematically under-serves the populations with the least existing access to care. If disparate performance is found, the model must either be corrected or its outputs must be adjusted with explicit equity constraints. The technical efficiency of the model does not override this obligation.

**Do not prioritize**

Resources directed at general behavioral promotion — campaigns encouraging handwashing or social distancing as a proxy for vaccine-receptive behavior — are unlikely to improve vaccination rates meaningfully. The data shows these behaviors are uncorrelated with vaccination uptake in any actionable way.

---

## What This Analysis Demonstrates

This project is structured as a complete data science workflow: exploratory analysis to understand the data and surface findings, preprocessing and feature engineering to translate those findings into a model-ready form, baseline and advanced modeling to produce a calibrated probability output, and evaluation to understand where the model performs and where it does not.

The technical output is a trained model and a set of probability scores. The business output is a prioritized targeting framework and a recommendation about where to spend campaign resources and why.

The distinction between those two outputs is the distinction between an analyst and a data scientist who drives decisions. The model is a means. The recommendation is the end. Every analytical choice made in the five notebooks preceding this write-up — which features to engineer, how to handle missing data, which metric to optimize, which threshold to examine — was made in service of that recommendation, not in service of the model itself.

That is the posture any data science project should take, regardless of domain. Start with the decision. Build backward to the data.

---

*Full technical documentation, code, and reproducible notebooks are available in the project repository. Model outputs should be validated against current population data before any operational deployment.*
