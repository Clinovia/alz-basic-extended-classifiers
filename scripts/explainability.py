import argparse
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

def explain_model(model_path, data_csv, scaler_path, feature_file, save_dir="explainability_outputs"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("🔹 Loading model and preprocessing assets...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(feature_file)

    print(f"✅ Loaded model: {model_path}")
    print(f"✅ Loaded {len(features)} features from {feature_file}")

    # Load dataset and match columns
    df = pd.read_csv(data_csv, low_memory=False)
    df = df[[f for f in features if f in df.columns]].copy()
    print(f"✅ Loaded dataset with {len(df)} records and {df.shape[1]} features")

    # Scale numeric columns
    num_cols = [c for c in df.columns if c not in ["PTGENDER", "APOE4"]]
    df[num_cols] = scaler.transform(df[num_cols])

    # Convert to numpy for SHAP
    X = df.values

    # SHAP explainability
    print("🔹 Computing SHAP values (this may take a few minutes)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Plot overall importance (global)
    print("🔹 Generating SHAP summary plots...")
    plt.figure()
    shap.summary_plot(shap_values, df, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary_bar.png", dpi=300)
    plt.close()

    # Standard summary plot
    shap.summary_plot(shap_values, df, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary_scatter.png", dpi=300)
    plt.close()

    print(f"✅ SHAP plots saved to {save_dir}/")

    # Optional: per-class breakdown for multi-class models
    if isinstance(shap_values, list):
        print("🔹 Generating per-class SHAP plots...")
        for i, class_name in enumerate(["CN", "MCI", "AD"]):
            plt.figure()
            shap.summary_plot(shap_values[i], df, show=False)
            plt.title(f"SHAP Summary for Class: {class_name}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/shap_class_{class_name}.png", dpi=300)
            plt.close()
        print("✅ Per-class plots saved.")

    print("✨ Explainability analysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Explainability with SHAP")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pkl file")
    parser.add_argument("--data-csv", type=str, required=True, help="Path to ADNIMERGE.csv or test dataset")
    parser.add_argument("--scaler", type=str, required=True, help="Path to saved scaler .pkl file")
    parser.add_argument("--features", type=str, default=None, help="Path to feature list (auto-detect if None)")
    args = parser.parse_args()

    # Auto-detect feature list based on model name
    if args.features is None:
        if "basic" in args.model:
            args.features = "models/basic_features.pkl"
        elif "advanced" in args.model:
            args.features = "models/advanced_features.pkl"
        else:
            raise ValueError("Could not infer features file — please specify --features manually.")

    explain_model(
        model_path=args.model,
        data_csv=args.data_csv,
        scaler_path=args.scaler,
        feature_file=args.features,
    )
