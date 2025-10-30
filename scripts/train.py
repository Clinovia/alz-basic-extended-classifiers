import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from preprocessing import load_and_preprocess_adni

# Load processed data
(basic_train, basic_test, yb_train, yb_test), (adv_train, adv_test, ya_train, ya_test) = load_and_preprocess_adni("data/ADNIMERGE.csv")

# ===== BASIC MODEL =====
basic_model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42
)
basic_model.fit(basic_train, yb_train)
yb_pred = basic_model.predict(basic_test)
print("=== BASIC MODEL PERFORMANCE ===")
print(classification_report(yb_test, yb_pred, target_names=["CN", "MCI", "AD"]))
joblib.dump(basic_model, "models/basic_model.pkl")

# ===== ADVANCED MODEL =====
adv_model = XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42
)
adv_model.fit(adv_train, ya_train)
ya_pred = adv_model.predict(adv_test)
print("=== ADVANCED MODEL PERFORMANCE ===")
print(classification_report(ya_test, ya_pred, target_names=["CN", "MCI", "AD"]))
joblib.dump(adv_model, "models/advanced_model.pkl")

