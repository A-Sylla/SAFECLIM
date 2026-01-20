# =============================================================================
# BAYESIAN HYPERPARAMETER OPTIMIZATION – MODEL COMPARISON
# Handling class imbalance:
#   • No oversampling
#   • SMOTE
#   • Native class weighting
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, recall_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from skopt.space import Real, Integer, Categorical

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

RANDOM_STATE = 100
N_ITER_BAYES = 100      # <-- amélioration majeure vs n_iter=1
OUTER_CV = 5
INNER_CV = 5

os.makedirs("saved_models", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# 1. Bayesian search spaces
# =============================================================================

bayes_spaces = {
    "Logistic Regression": {
        "C": Real(1e-4, 1e2, prior="log-uniform"),
        "penalty": Categorical(["l1", "l2"]),
        "solver": Categorical(["liblinear"]),
    },
    "Random Forest": {
        "n_estimators": Integer(50, 1000),
        "max_depth": Integer(3, 20),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", None]),
    },
    "XGBoost": {
        "n_estimators": Integer(50, 1000),
        "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
        "max_depth": Integer(3, 10),
        "min_child_weight": Integer(1, 10),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
        "gamma": Real(0.0, 1.0),
        "reg_alpha": Real(1e-9, 1.0, prior="log-uniform"),
        "reg_lambda": Real(1e-9, 1.0, prior="log-uniform"),
    },
    "Extra Trees": {
        "n_estimators": Integer(50, 1000),
        "max_depth": Integer(3, 20),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", None]),
        "bootstrap": Categorical([True, False]),
    },
}

# =============================================================================
# 2. Utility functions
# =============================================================================

def get_lag_features(df, pattern="lag"):
    """Return all lagged feature columns."""
    return [c for c in df.columns if pattern in c]

def build_preprocessor():
    """Global preprocessing pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

def safe_specificity(y_true, y_pred):
    """Compute specificity safely."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def format_search_range(space):
    """ Build Table hyperparameters
    """
    if isinstance(space, Real):
        if space.prior == "log-uniform":
            return f"[{space.low:.0e}, {space.high:.0e}] (log)"
        return f"[{space.low:.3g}, {space.high:.3g}]"
    elif isinstance(space, Integer):
        return f"[{space.low}, {space.high}]"
    elif isinstance(space, Categorical):
        return "[" + ", ".join("None" if v is None else str(v) for v in space.categories) + "]"
    return str(space)

def build_lags(climat_df, salmon_df, max_days=42, interval=7):
    """ 
    Constructs lagged variables for windows ranging from 7 to 42 days.
    """
    results = []
    for dep in salmon_df['Departement'].unique():
        clim_dep = climat_df[climat_df['Departement'] == dep].sort_values("Date").set_index("Date")
        salm_dep = salmon_df[salmon_df['Departement'] == dep]

        for _, row in salm_dep.iterrows():
            date_prel = row['Date']
            feats = row.to_dict()

            for lag in range(0, max_days, interval):
                start = date_prel - pd.Timedelta(days=lag + interval)
                end   = date_prel - pd.Timedelta(days=lag + 1)
                window = clim_dep.loc[start:end]

                if window.empty:
                    continue

                w = lag // interval

                # Temperature 
                feats[f'Nb_Extreme_Hot_Days_lag{lag//interval}'] = (window['Temperature_Maximale_C'] >= 30).sum()
                feats[f'Nb_Hot_Days_lag{lag//interval}'] = ((window['Temperature_Maximale_C'] >= 20) & (window['Temperature_Maximale_C'] < 30)).sum()
                feats[f'Nb_Moderate_Days_lag{lag//interval}'] = ((window['Temperature_Maximale_C'] >= 10) & (window['Temperature_Maximale_C'] < 20)).sum()
                feats[f'Nb_Cold_Days_lag{lag//interval}'] = (window['Temperature_Maximale_C'] < 10).sum()

                # Precipitations 
                feats[f'Nb_No_Rain_lag{w}']     = (window['Precipitations_24h_mm'] == 0).sum()
                feats[f'Nb_Light_Rain_lag{w}']  = (window['Precipitations_24h_mm'] < 0.04).sum()
                feats[f'Nb_Moderate_Rain_lag{w}'] = ((window['Precipitations_24h_mm'] >= 0.04) & (window['Precipitations_24h_mm'] <= 0.3)).sum()
                feats[f'Nb_High_Rain_lag{w}']   = (window['Precipitations_24h_mm'] > 3).sum()

                # Humidity
                feats[f'Nb_Very_High_Humidity_lag{w}'] = (window['Humidite_Moyenne_pct'] >= 90).sum()
                feats[f'Nb_High_Humidity_lag{w}']      = ((window['Humidite_Moyenne_pct'] >= 80) & (window['Humidite_Moyenne_pct'] < 90)).sum()
                feats[f'Nb_Moderate_Humidity_lag{w}']  = ((window['Humidite_Moyenne_pct'] >= 70) & (window['Humidite_Moyenne_pct'] < 80)).sum()
                feats[f'Nb_Low_Humidity_lag{w}']       = (window['Humidite_Moyenne_pct'] < 70).sum()

            results.append(feats)

    return pd.DataFrame(results)

# =============================================================================
# 3. Data loading
# =============================================================================

# NOTE: Les données brutes (microbiologiques et climatiques) ne sont pas incluses dans ce dépôt pour des raisons de confidentialité.
# L'utilisateur doit fournir ses propres fichiers et définir les chemins ci-dessous (ou via un fichier de configuration).
#
# Exemple :
# SALMONELLA_PATH = "data/salmonella_data.xlsx"
# CLIMATE_PATH    = "data/climate_data.xlsx"

# salmonella = pd.read_excel(SALMONELLA_PATH)  
# climat = pd.read_excel(CLIMATE_PATH)

# dataset_ish = build_lag_features(climat, salmonella)
# dataset_ish.to_excel("dataset_final_with_lags.xlsx", index=False)


dataset_ish = pd.read_excel(dataset_final_prelevements_with_lags.xlsx")

df = dataset_ish[dataset_ish["climate"].isin(["oceanic", "altered oceanic"])].copy()

X = df[get_lag_features(df)]
y = df["positif"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# =============================================================================
# 4. Models
# =============================================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=RANDOM_STATE, verbosity=0),
}

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0

models_weighted = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
}

# =============================================================================
# 5. Nested CV evaluation
# =============================================================================

def evaluate_scenario(X_train, y_train, X_test, y_test, models_dict, scenario_name):

    preproc = build_preprocessor()
    X_train_p = preproc.fit_transform(X_train)
    X_test_p = preproc.transform(X_test)

    outer_cv = StratifiedKFold(
        n_splits=OUTER_CV, shuffle=True, random_state=RANDOM_STATE
    )

    results = {}
    roc_data = {}

    for model_name, base_model in models_dict.items():
        print(f"\n→ {model_name} [{scenario_name}]")

        metrics = {
            "auc": [], "balanced_acc": [], "f1": [], "recall": [], "specificity": []
        }
        best_params = []

        for fold, (tr, val) in enumerate(outer_cv.split(X_train_p, y_train)):
            X_tr, X_val = X_train_p[tr], X_train_p[val]
            y_tr, y_val = y_train.iloc[tr], y_train.iloc[val]

            if scenario_name == "SMOTE":
                X_tr, y_tr = SMOTE(random_state=RANDOM_STATE + fold).fit_resample(X_tr, y_tr)

            opt = BayesSearchCV(
                base_model,
                bayes_spaces.get(model_name, {}),
                n_iter=N_ITER_BAYES,
                cv=INNER_CV,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=0
            )

            opt.fit(X_tr, y_tr)
            model = opt.best_estimator_
            best_params.append(opt.best_params_)

            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            metrics["auc"].append(roc_auc_score(y_val, y_proba))
            metrics["balanced_acc"].append(balanced_accuracy_score(y_val, y_pred))
            metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            metrics["specificity"].append(safe_specificity(y_val, y_pred))

        final_params = dict(Counter(tuple(sorted(p.items())) for p in best_params).most_common(1)[0][0])

        final_model = base_model.__class__(**{**base_model.get_params(), **final_params})

        X_tr_final, y_tr_final = X_train_p, y_train
        if scenario_name == "SMOTE":
            X_tr_final, y_tr_final = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr_final, y_tr_final)

        final_model.fit(X_tr_final, y_tr_final)

        y_test_proba = final_model.predict_proba(X_test_p)[:, 1]
        y_test_pred = (y_test_proba >= 0.5).astype(int)

        results[model_name] = {
            "Model": model_name,
            "Scenario": scenario_name,
            "AUC_CV": np.mean(metrics["auc"]),
            "Balanced_Acc_CV": np.mean(metrics["balanced_acc"]),
            "F1_CV": np.mean(metrics["f1"]),
            "Recall_CV": np.mean(metrics["recall"]),
            "Specificity_CV": np.mean(metrics["specificity"]),
            "AUC_Test": roc_auc_score(y_test, y_test_proba),
            "Balanced_Acc_Test": balanced_accuracy_score(y_test, y_test_pred),
            "F1_Test": f1_score(y_test, y_test_pred, zero_division=0),
            "Recall_Test": recall_score(y_test, y_test_pred, zero_division=0),
            "Specificity_Test": safe_specificity(y_test, y_test_pred),
            "Model_Object": final_model
        }

        roc_data[f"{model_name} ({scenario_name})"] = (y_test, y_test_proba)

    return pd.DataFrame(results).T, roc_data, preproc

# =============================================================================
# 6. Run scenarios
# =============================================================================

all_results, all_roc, preprocessors = [], {}, {}

scenarios = [
    ("No SMOTE", models),
    ("SMOTE", models),
    ("Native Weighting", models_weighted),
]

for name, model_dict in scenarios:
    print("\n" + "=" * 100)
    print(f" SCENARIO : {name} ".center(100))
    print("=" * 100)

    df_tmp, roc_tmp, preproc = evaluate_scenario(X_train, y_train, X_test, y_test, model_dict, name)

    all_results.append(df_tmp)
    all_roc.update(roc_tmp)
    preprocessors[name] = preproc

df_comp = pd.concat(all_results, ignore_index=True)

# =============================================================================
# 7. Save best model
# =============================================================================

df_comp["Combined_Score"] = 0.5 * df_comp["AUC_Test"] + 0.5 * df_comp["Recall_Test"]
best = df_comp.sort_values("Combined_Score", ascending=False).iloc[0]

model_name = best["Model"].lower().replace(" ", "_")
scenario_name = best["Scenario"].lower().replace(" ", "_")

joblib.dump(best["Model_Object"], f"saved_models/best_model_{model_name}_{scenario_name}.pkl")
joblib.dump(preprocessors[best["Scenario"]], f"saved_models/preprocessor_{scenario_name}.pkl")

print("\n✓ Best model saved successfully")

# =============================================================================
# 8.1 TABLE 2 – HYPERPARAMETER SEARCH SPACES AND OPTIMAL VALUES
# =============================================================================

model_order = ["Extra Trees", "Random Forest", "XGBoost", "Logistic Regression"]

rows = []

for model_name in model_order:

    if model_name not in bayes_spaces:
        continue

    search_space = bayes_spaces[model_name]

    # Best-performing instance of this model (across scenarios)
    subset = df_comp[df_comp["Model"] == model_name]

    if subset.empty:
        best_params = {}
    else:
        best_row = subset.loc[subset["AUC_Test"].idxmax()]
        best_params = best_row["Model_Object"].get_params()

    for i, (param, space) in enumerate(search_space.items()):
        rows.append({
            "Model": model_name if i == 0 else "",
            "Hyperparameter": param,
            "Search range": format_search_range(space),
            "Optimal value": (
                f"{best_params[param]:.4g}"
                if param in best_params and isinstance(best_params[param], float)
                else str(best_params.get(param, "-"))
            )
        })

df_hyperparams = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------------

print("\n" + "=" * 110)
print(" TABLE 2 – HYPERPARAMETER SEARCH SPACES AND OPTIMAL VALUES ".center(110))
print("=" * 110)
print(df_hyperparams.to_string(index=False))

# -----------------------------------------------------------------------------
# Save (publication-ready)
# -----------------------------------------------------------------------------

df_hyperparams.to_csv("table2_hyperparameters.csv", index=False)
print("\n→ Saved: table2_hyperparameters.csv")



# =============================================================================
# 8.2 Performance summary – TRAIN (CV)
# =============================================================================

train_cols = ["Model", "Scenario", "AUC_CV", "Balanced_Acc_CV", "F1_CV", "Recall_CV", "Specificity_CV"]

df_train_summary = df_comp[train_cols].copy()

df_train_summary = df_train_summary.rename(columns={
    "AUC_CV": "AUC (CV)", "Balanced_Acc_CV": "Balanced Acc. (CV)", "F1_CV": "F1 (CV)", "Recall_CV": "Recall (CV)",
    "Specificity_CV": "Specificity (CV)"
})

df_train_summary.iloc[:, 2:] = df_train_summary.iloc[:, 2:].round(4)

df_train_summary = df_train_summary.sort_values(by=["Scenario", "AUC (CV)"], ascending=[True, False]).reset_index(drop=True)

print("\n" + "=" * 100)
print(" PERFORMANCE SUMMARY – TRAIN SET (NESTED CV) ".center(100))
print("=" * 100)
print(df_train_summary.to_string(index=False))

df_train_summary.to_csv("performance_summary_train_cv_by_scenario.csv", index=False)

print("\n→ Saved: performance_summary_train_cv_by_scenario.csv")

# =============================================================================
# 8.3 Performance summary – TEST
# =============================================================================

test_cols = ["Model", "Scenario", "AUC_Test", "Balanced_Acc_Test", "F1_Test", "Recall_Test", "Specificity_Test"]

df_test_summary = df_comp[test_cols].copy()

df_test_summary = df_test_summary.rename(columns={
    "AUC_Test": "AUC", "Balanced_Acc_Test": "Balanced Acc.", "F1_Test": "F1", "Recall_Test": "Recall", "Specificity_Test": "Specificity"})

df_test_summary.iloc[:, 2:] = df_test_summary.iloc[:, 2:].round(4)

df_test_summary = df_test_summary.sort_values(by=["Scenario", "AUC"], ascending=[True, False]).reset_index(drop=True)

print("\n" + "=" * 100)
print(" PERFORMANCE SUMMARY – TEST SET ".center(100))
print("=" * 100)
print(df_test_summary.to_string(index=False))

df_test_summary.to_csv("performance_summary_test_by_scenario.csv", index=False)

print("\n→ Saved: performance_summary_test_by_scenario.csv")


# =============================================================================
# 8.4 Confusion matrices – grid (models × scenarios)
# =============================================================================

available_models = df_comp["Model"].unique().tolist()
available_scenarios = sorted(df_comp["Scenario"].unique())

model_order = (
    df_comp.groupby("Model")["AUC_Test"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

n_models = len(model_order)
n_scenarios = len(available_scenarios)

fig, axes = plt.subplots(n_models, n_scenarios, figsize=(5 * n_scenarios + 2, 5 * n_models))

if n_models == 1:
    axes = axes.reshape(1, -1)
elif n_scenarios == 1:
    axes = axes.reshape(-1, 1)

# Preprocess test set once
best_preproc = preprocessors[best["Scenario"]]
X_test_proc = pd.DataFrame( best_preproc.transform(X_test), columns=X_test.columns, index=X_test.index)

for i, model_name in enumerate(model_order):
    for j, scenario in enumerate(available_scenarios):
        ax = axes[i, j]

        match = df_comp[
            (df_comp["Model"] == model_name) &
            (df_comp["Scenario"] == scenario)
        ]

        if match.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Not trained", ha="center", va="center", fontsize=12)
            continue

        model = match.iloc[0]["Model_Object"]
        auc = match.iloc[0]["AUC_Test"]

        y_pred = model.predict(X_test_proc)
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)

        ax.set_title(f"{scenario}\nAUC={auc:.3f}", fontsize=11)
        ax.grid(False)

    axes[i, 0].set_ylabel(
        model_name, fontsize=13, fontweight="bold", labelpad=15
    )

plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08, wspace=0.4, hspace=0.5)

plt.savefig("confusion_matrices_by_model_and_scenario.png", dpi=300, bbox_inches="tight")
plt.show()

print("→ Confusion matrices saved: confusion_matrices_by_model_and_scenario.png")



