import joblib
import pandas as pd

# Load scaler
scaler = joblib.load("scaler_advanced.pkl")

# If it's a plain StandardScaler
if hasattr(scaler, "feature_names_in_"):
    features = list(scaler.feature_names_in_)
else:
    print("No feature names stored in scaler. We'll use CSV columns instead.")
    df = pd.read_csv("adni_merge_features_renamed.csv")
    features = df.columns.tolist()

print("Advanced model features:")
print(features)
