"""
Split → Train: 1596, Test: 400
Split → Train: 1596, Test: 400
Saved scaler → models/scaler_basic.pkl
Saved scaler → models/scaler_advanced.pkl
Saved 14 basic features
Saved 40 advanced features

================ BASIC FEATURE SET ================

LogisticRegression (Basic) Classification Report:
              precision    recall  f1-score   support

          CN       0.55      0.87      0.67       109
         MCI       0.81      0.55      0.65       209
          AD       0.80      0.85      0.83        82

    accuracy                           0.70       400
   macro avg       0.72      0.76      0.72       400
weighted avg       0.74      0.70      0.69       400

Sensitivity & Specificity:
CN: Sensitivity=0.872, Specificity=0.732
MCI: Sensitivity=0.545, Specificity=0.864
AD: Sensitivity=0.854, Specificity=0.947

Saved model → models/basic_logisticregression.pkl

🔹 Training RandomForest (Basic)...
RandomForest (Basic) Classification Report:
              precision    recall  f1-score   support

          CN       0.65      0.59      0.62       109
         MCI       0.73      0.79      0.76       209
          AD       0.88      0.80      0.84        82

    accuracy                           0.74       400
   macro avg       0.75      0.73      0.74       400
weighted avg       0.74      0.74      0.74       400

Sensitivity & Specificity:
CN: Sensitivity=0.587, Specificity=0.880
MCI: Sensitivity=0.789, Specificity=0.681
AD: Sensitivity=0.805, Specificity=0.972

Saved model → models/basic_randomforest.pkl

🔹 Training XGBoost (Basic)...
XGBoost (Basic) Classification Report:
              precision    recall  f1-score   support

          CN       0.61      0.62      0.62       109
         MCI       0.72      0.74      0.73       209
          AD       0.86      0.78      0.82        82

    accuracy                           0.72       400
   macro avg       0.73      0.72      0.72       400
weighted avg       0.72      0.72      0.72       400

Sensitivity & Specificity:
CN: Sensitivity=0.624, Specificity=0.849
MCI: Sensitivity=0.742, Specificity=0.691
AD: Sensitivity=0.780, Specificity=0.969

Saved model → models/basic_xgboost.pkl
"""

import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import load_and_preprocess_adni

# ============================================================
# LOAD DATA (BASIC ONLY)
# ============================================================

(basic_train, basic_test, yb_train, yb_test), _ = load_and_preprocess_adni(
    "data/ADNIMERGE.csv"
)

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
# MODELS
# ============================================================
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
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

print("\n================ BASIC FEATURE SET ================")
for name, model in models.items():
    run_and_report(
        model,
        f"{name} (Basic)",
        basic_train,
        basic_test,
        yb_train,
        yb_test,
        f"models/basic_{name.lower()}.pkl",
    )
