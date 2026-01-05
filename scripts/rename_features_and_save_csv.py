import pandas as pd
import os

def rename_adni_columns(input_csv: str, output_csv: str):
    # Load data
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df)} records from {input_csv}")

    # Drop "_bl" suffix from columns
    df = df.rename(columns=lambda x: x.replace("_bl", ""))

    # Save to new file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved renamed dataset to {output_csv}")


if __name__ == "__main__":
    input_path = "data/ADNIMERGE.csv"
    output_path = "data/adni_merge_renamed_features.csv"
    rename_adni_columns(input_path, output_path)
