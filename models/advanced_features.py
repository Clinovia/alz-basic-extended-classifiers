import joblib

# Load model saved with joblib
model = joblib.load("advanced_model.pkl")

# Check features if possible
if hasattr(model, "feature_names_in_"):
    print("Features used by model:")
    print(model.feature_names_in_)
else:
    print("Model does not have feature_names_in_ attribute")
