# Climate-Driven Risk of Salmonella Contamination in Poultry

This repository contains the code and analysis pipeline used to assess the influence of climatic conditions on the risk of *Salmonella* contamination in poultry production systems in France, using machine learning approaches.

The study focuses on identifying delayed (lagged) effects of temperature, humidity, and precipitation on *Salmonella* detection, with the objective of supporting climate-informed food safety risk assessment and early warning strategies.

---

## 📌 Project Overview

- **Objective:**  
  To evaluate how climatic variability influences *Salmonella* contamination in poultry using data-driven, non-linear modeling approaches.

- **Approach:**  
  - Integration of meteorological variables classified by intensity (temperature, humidity, precipitation)
  - Analysis of multiple temporal lags (0–5 weeks before sampling)
  - Comparison of several machine learning classifiers
  - Model interpretation using SHAP (Shapley Additive Explanations)

- **Study area:**  
  France (oceanic and altered oceanic climate zones)

---

## 📊 Methods

- **Models evaluated:**
  - XGBoost
  - Random Forest
  - Extra Trees
  - Logistic Regression

- **Class imbalance handling:**
  - Native class weighting
  - SMOTE-based strategies
  - Baseline (no rebalancing)

- **Evaluation metrics:**
  - AUC-ROC
  - Recall (sensitivity)
  - Balanced Accuracy
  - Confusion matrices

- **Interpretability:**
  - SHAP values for feature importance and effect direction

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                # Raw input data (not shared if confidential)
│   ├── processed/          # Cleaned and feature-engineered datasets
│
├── scripts/
│   ├── preprocessing/      # Data cleaning and feature engineering
│   ├── modeling/           # Model training and evaluation
│   ├── interpretation/     # SHAP analysis and visualization
│
├── figures/
│   ├── roc_curves/          # ROC curves for all lags
│   ├── confusion_matrices/ # Confusion matrices
│   ├── shap/               # SHAP summary plots
│
├── results/
│   ├── metrics_tables/     # Performance metrics (CSV)
│
├── environment.yml         # Conda environment (optional)
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md
