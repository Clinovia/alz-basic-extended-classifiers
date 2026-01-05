"""
Numeric columns (21): ['AGE', 'MMSE', 'FAQ', 'PTEDUCAT', 'APOE4', 'RAVLT_immediate', 'MOCA', 'ADAS13', 'Hippocampus', 'Ventricles', 'WholeBrain', 'Entorhinal', 'FDG', 'AV45', 'PIB', 'FBB', 'ABETA', 'TAU', 'PTAU', 'mPACCdigit', 'mPACCtrailsB']
Split → Train: 1596, Test: 400
Split → Train: 1596, Test: 400
Saved scaler → models/scaler_basic.pkl
Saved scaler → models/scaler_advanced.pkl
Saved 9 basic features
Saved 22 advanced features

🔍 Auditing adv_train
Shape: (1596, 22)
✅ No duplicate columns

🔍 Auditing adv_test
Shape: (400, 22)
✅ No duplicate columns

================ ADVANCED FEATURE SET ================

🔹 Training LogisticRegression (Advanced)...
LogisticRegression (Advanced) Classification Report:
              precision    recall  f1-score   support

          CN       0.60      0.89      0.72       109
         MCI       0.84      0.62      0.71       209
          AD       0.82      0.84      0.83        82

    accuracy                           0.74       400
   macro avg       0.75      0.78      0.75       400
weighted avg       0.77      0.74      0.74       400

Sensitivity & Specificity:
CN: Sensitivity=0.890, Specificity=0.780
MCI: Sensitivity=0.622, Specificity=0.869
AD: Sensitivity=0.841, Specificity=0.953

Saved model → models/advanced_logisticregression.pkl

🔹 Training RandomForest (Advanced)...
RandomForest (Advanced) Classification Report:
              precision    recall  f1-score   support

          CN       0.69      0.63      0.66       109
         MCI       0.74      0.81      0.77       209
          AD       0.87      0.76      0.81        82

    accuracy                           0.75       400
   macro avg       0.77      0.73      0.75       400
weighted avg       0.75      0.75      0.75       400

Sensitivity & Specificity:
CN: Sensitivity=0.633, Specificity=0.893
MCI: Sensitivity=0.809, Specificity=0.686
AD: Sensitivity=0.756, Specificity=0.972

Saved model → models/advanced_randomforest.pkl

🔹 Training XGBoost (Advanced)...
XGBoost (Advanced) Classification Report:
              precision    recall  f1-score   support

          CN       0.68      0.70      0.69       109
         MCI       0.77      0.78      0.78       209
          AD       0.88      0.80      0.84        82

    accuracy                           0.77       400
   macro avg       0.78      0.76      0.77       400
weighted avg       0.77      0.77      0.77       400

Sensitivity & Specificity:
CN: Sensitivity=0.697, Specificity=0.876
MCI: Sensitivity=0.785, Specificity=0.743
AD: Sensitivity=0.805, Specificity=0.972
"""


import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import load_and_preprocess_adni

# ============================================================
# LOAD DATA (ADVANCED ONLY)
# ============================================================

_, (adv_train, adv_test, ya_train, ya_test) = load_and_preprocess_adni(
    "data/ADNIMERGE.csv"
)

# ============================================================
# DEBUG: COLUMN INTEGRITY CHECKS
# ============================================================

def audit_dataframe(X, name):
    print(f"\n🔍 Auditing {name}")
    print(f"Shape: {X.shape}")

    dup_cols = X.columns[X.columns.duplicated()].tolist()
    if dup_cols:
        print(f"❌ Duplicate columns detected ({len(dup_cols)}):")
        print(dup_cols)
    else:
        print("✅ No duplicate columns")

    bad_types = [c for c in X.columns if isinstance(X[c], pd.DataFrame)]
    if bad_types:
        raise ValueError(
            f"Columns returning DataFrame instead of Series: {bad_types}"
        )

    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        print(f"⚠️ Non-numeric columns ({len(non_numeric)}): {non_numeric}")

audit_dataframe(adv_train, "adv_train")
audit_dataframe(adv_test, "adv_test")

# Hard fail if duplicates exist
if adv_train.columns.duplicated().any():
    raise RuntimeError("Duplicate columns detected — fix preprocessing before training.")

# ============================================================
# HELPERS
# ============================================================

def print_sensitivity_specificity(y_true, y_pred, labels=["CN", "MCI", "AD"]):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        print(f"{label}: Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}")
    print()


def run_and_report(model, model_name, X_train, X_test, y_train, y_test, save_path):
    print(f"\n🔹 Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["CN", "MCI", "AD"]))
    print("Sensitivity & Specificity:")
    print_sensitivity_specificity(y_test, y_pred)

    joblib.dump(model, save_path)
    print(f"Saved model → {save_path}")

# ============================================================
# MODELS (INCLUDING XGBOOST)
# ============================================================

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    ),
}

# ============================================================
# RUN
# ============================================================

print("\n================ ADVANCED FEATURE SET ================")
for name, model in models.items():
    run_and_report(
        model,
        f"{name} (Advanced)",
        adv_train,
        adv_test,
        ya_train,
        ya_test,
        f"models/advanced_{name.lower()}.pkl",
    )
