# Climate-Driven Risk of *Salmonella* Contamination in Poultry

This repository contains the analysis pipeline and code used to investigate the influence of climatic conditions on the risk of *Salmonella* contamination in poultry production systems in France, leveraging machine learning methods.

The study focuses on detecting **lagged effects** of temperature, humidity, and precipitation on *Salmonella* detection, supporting **climate-informed food safety risk assessment** and **early warning strategies**.

---

## Project Overview

* **Goal:**
  Quantify the impact of climatic variability on *Salmonella* contamination in poultry using data-driven, non-linear models.

* **Key Features:**

  * Incorporates meteorological variables (temperature, humidity, precipitation) with intensity-based classification
  * Evaluates multiple temporal lags (0–5 weeks before sampling)
  * Compares several machine learning classifiers
  * Interprets models using **SHAP (Shapley Additive Explanations)**

* **Geographic focus:**
  France – oceanic and altered oceanic climate zones

---

## Scripts

* **`models_comparison.py`**

  * Compares multiple machine learning classifiers (XGBoost, Random Forest, Extra Trees, Logistic Regression)
  * Handles class imbalance using native weighting, SMOTE-based strategies, and baseline scenarios
  * Performs hyperparameter optimization via Bayesian search
  * Outputs performance metrics, ROC curves, confusion matrices, and feature importance (SHAP)

* **`lag_comparison.py`**

  * Evaluates the predictive performance of the best model across individual temporal lags
  * Generates lag-specific ROC curves and confusion matrices
  * Performs SHAP analysis on the best-performing lag
  * Saves visualizations for reporting and interpretation

---

## Methods

* **Models Evaluated:**

  * XGBoost
  * Random Forest
  * Extra Trees
  * Logistic Regression

* **Class Imbalance Strategies:**

  * Native class weighting
  * SMOTE-based oversampling
  * Baseline (no SMOTE)

* **Evaluation Metrics:**

  * Area Under the ROC Curve (AUC-ROC)
  * Recall / Sensitivity
  * Balanced Accuracy
  * Confusion Matrices

* **Interpretability:**

  * SHAP values for global and lag-specific feature importance
  * Beeswarm plots for top predictors

---

## Reproducibility and Data

* The raw surveillance data are **confidential and not publicly available**.
* The repository provides a **fully reproducible pipeline**: users can run the scripts with their own data formatted according to the expected structure.
* All hyperparameter search, model training, evaluation, and SHAP analysis steps are included.

---

## Quickstart / Usage

To reproduce the analysis pipeline, follow these steps:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages include:

* `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imblearn`, `scikit-optimize`
* `matplotlib`, `seaborn`
* `shap`, `joblib`

---

### 2. Prepare your data

* Provide your own **Salmonella surveillance** and **climatic datasets**.
* Ensure datasets include:

  * `Date` and `Departement` columns
  * Microbiological outcomes (e.g., `positif`)
  * Meteorological features: `Temperature_Maximale_C`, `Precipitations_24h_mm`, `Humidite_Moyenne_pct`
* Use the preprocessing pipeline in `models_comparison.py` to build lagged features if needed.

---

### 3. Run model comparison

```bash
python models_comparison.py
```

This script will:

* Compare multiple classifiers
* Perform Bayesian hyperparameter optimization
* Handle class imbalance with different strategies
* Output:

  * Performance metrics (train & test)
  * ROC curves & confusion matrices
  * SHAP feature importance plots
* Save the **best model** and **preprocessor** in `saved_models/`

---

### 4. Run lag-specific analysis

```bash
python lag_comparison.py
```

This script will:

* Load the best model and preprocessor saved by `models_comparison.py`
* Evaluate predictive performance for **individual lags**
* Generate:

  * ROC curves per lag
  * Confusion matrices per lag
  * SHAP beeswarm plots for the best-performing lag
* Save figures to the working directory for reporting

---

### 5. Outputs

Key outputs are saved in `saved_models/` and as figure files:

* `best_model.pkl` – trained model object
* `preprocessor.pkl` – fitted preprocessing pipeline
* `roc_curves_all_lags.png` – combined ROC curves per lag
* `confusion_matrices_all_lags_combined.png` – confusion matrices per lag
* `shap_beeswarm_<model>_lag<best_lag>.png` – SHAP feature importance plot
