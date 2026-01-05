import pandas as pd
import os

"""
Kept 2430 baseline (VISCODE='bl') records
"""

def filter_baseline_adni(input_csv: str, output_csv: str):
    # Load renamed data
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df)} total records from {input_csv}")

    # Filter baseline records (VISCODE == 'bl' or 'BL')
    if "VISCODE" not in df.columns:
        raise KeyError("Column 'VISCODE' not found in the dataset.")
        
    baseline_df = df[df["VISCODE"].str.lower() == "bl"].copy()
    print(f"Kept {len(baseline_df)} baseline (VISCODE='bl') records")

    # Save baseline dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    baseline_df.to_csv(output_csv, index=False)
    print(f"Saved baseline dataset to {output_csv}")


if __name__ == "__main__":
    input_path = "data/adni_merge_renamed_features.csv"
    output_path = "data/adni_merge_baseline.csv"
    filter_baseline_adni(input_path, output_path)
