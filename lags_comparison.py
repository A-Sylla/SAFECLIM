# =============================================================================
# PART 2 – LAG-BY-LAG SIMULATION USING BEST MODEL FROM PART 1
# =============================================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score, recall_score,
    matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    average_precision_score
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import joblib

# ------------------------------------------------------------
# LOAD BEST MODEL AND PREPROCESSOR
# ------------------------------------------------------------
best_model = joblib.load("saved_models/best_model.pkl")
preproc = joblib.load("saved_models/preprocessor.pkl")
best_model_name = type(best_model).__name__.replace("Classifier", "")
print(f"✔ Best model loaded: {best_model_name}")

# ------------------------------------------------------------
# LAG-BY-LAG SIMULATION
# ------------------------------------------------------------
print("\n" + "="*100)
print(" LAG-BY-LAG SIMULATION WITH FIXED BEST HYPERPARAMETERS ".center(100))
print("="*100 + "\n")

# Extract available lags automatically
lags = sorted(set(int(re.search(r"lag(\d+)", c).group(1)) 
                  for c in X.columns if re.search(r"lag(\d+)", c)))

lag_results = []
lag_roc_train = {}
lag_roc_test = {}


for lag in lags:
    print(f"\n--- Lag {lag} ---")
    lag_cols = [c for c in X.columns if f"lag{lag}" in c]

    # Extract train/test features for this lag
    X_train_lag = X_train_full[lag_cols].copy()
    X_test_lag = X_test[lag_cols].copy()

    # Preprocess data (fit per lag)
    preproc_lag = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_train_proc = preproc_lag.fit_transform(X_train_lag)
    X_test_proc = preproc_lag.transform(X_test_lag)

    # Refit model on this lag (same hyperparameters as best global model)
    model_lag = best_model.__class__(**best_model.get_params())
    model_lag.fit(X_train_proc, y_train_full)

    # Predict probabilities
    y_proba_train = model_lag.predict_proba(X_train_proc)[:, 1]
    y_proba_test  = model_lag.predict_proba(X_test_proc)[:, 1]
    y_pred_test   = (y_proba_test >= 0.5).astype(int)

    # Store for ROC plotting later
    lag_roc_train[lag] = (y_train_full, y_proba_train)
    lag_roc_test[lag]  = (y_test, y_proba_test)

    # Compute metrics
    auc_test = roc_auc_score(y_test, y_proba_test)
    pr_auc_test = average_precision_score(y_test, y_proba_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    lag_results.append({
        "Lag": lag,
        "AUC_Test": auc_test,
        "PR_AUC_Test": pr_auc_test,
        "Balanced_Acc_Test": bal_acc,
        "Recall_Test": recall,
        "Specificity_Test": specificity,
        "F1_Test": f1
    })

# ------------------------------------------------------------
# SUMMARY TABLE BY LAG
# ------------------------------------------------------------
lag_results = pd.DataFrame(lag_results)
lag_results['Combined_Score'] = 0.5 * lag_results['AUC_Test'] + 0.5 * lag_results['Recall_Test']
df_lag_results = lag_results.sort_values(by='PR_AUC_Test', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print(" FINAL LAG RESULTS (sorted by Test AUC) ".center(80))
print("="*80)
display(df_lag_results.round(4))

best_lag_row = df_lag_results.iloc[0]
best_lag = int(best_lag_row["Lag"])
print(f"\n✔ Best lag according to Test AUC: lag {best_lag} (AUC = {best_lag_row['AUC_Test']:.3f})")

# ------------------------------------------------------------
# SHAP ANALYSIS FOR BEST MODEL + BEST LAG
# ------------------------------------------------------------
print("\n" + "="*60)
print(" SHAP ANALYSIS FOR BEST MODEL + BEST LAG ".center(60))
print("="*60)

# Columns for best lag
lag_cols_best = [c for c in X.columns if f"lag{best_lag}" in c]

# Preprocessing
preproc_best_lag = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
X_train_proc_best = pd.DataFrame(preproc_best_lag.fit_transform(X_train_full[lag_cols_best]),
                                 columns=lag_cols_best, index=X_train_full.index)
X_test_proc_best = pd.DataFrame(preproc_best_lag.transform(X_test[lag_cols_best]),
                                columns=lag_cols_best, index=X_test.index)

# Refit best model on best lag
best_model_lag = best_model.__class__(**best_model.get_params())
best_model_lag.fit(X_train_proc_best, y_train_full)
X_shap = X_test_proc_best

# Choose explainer
if isinstance(best_model_lag, XGBClassifier):
    print("→ XGBoost detected: using PermutationExplainer")
    explainer = shap.PermutationExplainer(best_model_lag.predict_proba, X_shap)
    shap_values = explainer(X_shap)
elif isinstance(best_model_lag, (RandomForestClassifier, ExtraTreesClassifier)):
    print("→ Tree model detected: using TreeExplainer")
    explainer = shap.TreeExplainer(best_model_lag)
    shap_values = explainer(X_shap)
else:
    print("→ Generic explainer")
    explainer = shap.Explainer(best_model_lag, X_shap)
    shap_values = explainer(X_shap)

# Extract positive class SHAP
shap_values_pos = shap_values  #[..., 1] if shap_values.shape[-1] > 1 else shap_values

# Clean feature names
def clean_feature_name(name):
    return re.sub(r'(?:_?lag\d+)', '', name).strip('_ ')
clean_feature_names = [clean_feature_name(col) for col in X_shap.columns]

# Top 15 SHAP features
shap_abs = np.abs(shap_values_pos.values).mean(axis=0)
feature_importance_clean = (pd.DataFrame({"feature": clean_feature_names, "mean_abs_shap": shap_abs})
                            .sort_values("mean_abs_shap", ascending=True).head(15).reset_index(drop=True))
top_features = feature_importance_clean["feature"].tolist()
feature_idx = [clean_feature_names.index(f) for f in top_features]
X_shap_top = X_shap.iloc[:, feature_idx]
shap_values_top = shap_values_pos[:, feature_idx]

# Beeswarm plot
fig2 = plt.figure(figsize=(6,6))
shap.summary_plot(
    shap_values_top,
    X_shap_top,
    feature_names=top_features,
    sort=True,
    max_display=15,
    show=False
)
plt.xlabel("SHAP value\n(Impact on positive model output)", fontsize=11)
plt.title(f"Beeswarm\n{best_model_name} – Lag {best_lag}", fontsize=12)
plt.tight_layout()
filename_beeswarm = f"shap_beeswarm_{best_model_name}_lag{best_lag}.png"
fig2.savefig(filename_beeswarm, dpi=300, bbox_inches="tight", facecolor="white")
print(f"✔ Beeswarm saved as '{filename_beeswarm}'")
plt.show()

# ------------------------------------------------------------
# ROC CURVES FOR ALL LAGS
# ------------------------------------------------------------
print("\n" + "="*100)
print(" ROC CURVES PER LAG ".center(100))
print("="*100 + "\n")

cols = 2
n_lags = len(lag_roc_train)
rows = (n_lags + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6*cols,5*rows))
axes = axes.flatten() if n_lags > 1 else [axes]

for idx, lag in enumerate(sorted(lag_roc_train.keys())):
    ax = axes[idx]
    y_tr, s_tr = lag_roc_train[lag]
    y_te, s_te = lag_roc_test[lag]
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, s_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, s_te)
    auc_tr = roc_auc_score(y_tr, s_tr)
    auc_te = roc_auc_score(y_te, s_te)

    ax.plot(fpr_tr, tpr_tr, label=f"Train (AUC={auc_tr:.3f})", linestyle="--")
    ax.plot(fpr_te, tpr_te, label=f"Test (AUC={auc_te:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Lag {lag}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

for idx in range(n_lags, len(axes)):
    axes[idx].axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig("roc_curves_all_lags.png", dpi=300, bbox_inches='tight', facecolor='white')
print("✔ ROC curves per lag saved as 'roc_curves_all_lags.png'")
plt.show()

# ------------------------------------------------------------
# CONFUSION MATRICES FOR ALL LAGS
# ------------------------------------------------------------
print("\n" + "="*100)
print(" CONFUSION MATRICES PER LAG ".center(100))
print("="*100 + "\n")

cols = 3
rows = (n_lags + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5*cols,4.5*rows))
axes = axes.flatten() if n_lags > 1 else [axes]

plot_idx = 0
for lag in sorted(lags):
    lag_cols = [c for c in X.columns if f"lag{lag}" in c]
    if len(lag_cols) == 0:
        continue

    X_train_lag = X_train_full[lag_cols].copy()
    X_test_lag = X_test[lag_cols].copy()
    preproc_lag = Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())])
    X_train_proc = preproc_lag.fit_transform(X_train_lag)
    X_test_proc = preproc_lag.transform(X_test_lag)

    model_lag = best_model.__class__(**best_model.get_params())
    model_lag.fit(X_train_proc, y_train_full)

    y_pred_test = model_lag.predict(X_test_proc)
    cm = confusion_matrix(y_test, y_pred_test)

    ax = axes[plot_idx]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize":12})
    ax.set_title(f"Lag {lag}", fontsize=12, pad=10)
    plot_idx += 1

for idx in range(plot_idx, len(axes)):
    axes[idx].axis('off')

plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.savefig("confusion_matrices_all_lags_combined.png", dpi=300, bbox_inches='tight', facecolor='white')
print("✔ All confusion matrices per lag saved as 'confusion_matrices_all_lags_combined.png'")
plt.show()

