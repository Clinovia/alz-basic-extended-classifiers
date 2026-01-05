import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Utilities
# ============================================================

def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicated columns by taking the first non-null value row-wise.
    Guarantees unique column names and Series-only access.
    """
    if not df.columns.duplicated().any():
        return df

    print("⚠️ Resolving duplicated columns...")

    new_df = pd.DataFrame(index=df.index)

    for col in df.columns.unique():
        cols = df.loc[:, df.columns == col]

        if cols.shape[1] == 1:
            new_df[col] = cols.iloc[:, 0]
        else:
            print(f"⚠️ Collapsing duplicated column: {col}")
            new_df[col] = cols.bfill(axis=1).iloc[:, 0]

    assert not new_df.columns.duplicated().any(), "Duplicate columns remain!"
    return new_df


# ============================================================
# Main loader
# ============================================================

def load_and_preprocess_adni(csv_path: str, save_dir: str = "models"):
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # Load CSV
    # =========================
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} total records")

    # =========================
    # Keep baseline visits only
    # =========================
    if "VISCODE" in df.columns:
        df = df[df["VISCODE"].astype(str).str.lower() == "bl"].copy()
        print(f"Kept {len(df)} baseline (VISCODE='bl') records")

    # =========================
    # Unique patients
    # =========================
    if "RID" in df.columns:
        df = df.drop_duplicates(subset="RID")
        print(f"After removing duplicates by RID: {len(df)} unique patients")

    # =========================
    # Diagnosis handling
    # =========================
    if "DX_bl" not in df.columns:
        raise ValueError("DX_bl column not found")

    df = df.dropna(subset=["DX_bl"])
    df["DX_bl"] = df["DX_bl"].replace({"SMC": "MCI", "LMCI": "MCI"})

    label_map = {"CN": 0, "MCI": 1, "AD": 2}
    df["target"] = df["DX_bl"].map(label_map).astype("Int64")
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    # =========================
    # Rename *_bl → base names
    # =========================
    df = df.rename(columns=lambda c: c.replace("_bl", ""))
    renamed_path = os.path.join(save_dir, "adni_merge_features_renamed.csv")
    df.to_csv(renamed_path, index=False)
    print(f"Saved renamed dataset → {renamed_path}")

    # =========================
    # Feature definitions
    # =========================
    basic_features = [
        "AGE", "MMSE", "FAQ", "PTEDUCAT",
        "PTGENDER", "APOE4",
        "RAVLT_immediate", "MOCA", "ADAS13"
    ]

    advanced_features = basic_features + [
        "Hippocampus", "Ventricles", "WholeBrain", "Entorhinal",
        "FDG", "AV45", "PIB", "FBB",
        "ABETA", "TAU", "PTAU",
        "mPACCdigit", "mPACCtrailsB"
    ]

    # =========================
    # Collapse duplicated columns EARLY
    # =========================
    df = collapse_duplicate_columns(df)

    # =========================
    # Numeric coercion (ADNI-safe)
    # =========================
    numeric_cols = [
        c for c in advanced_features
        if c in df.columns and c not in ["PTGENDER"]
    ]

    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # =========================
    # Categorical encoding
    # =========================
    if "PTGENDER" in df.columns:
        df["PTGENDER"] = (
            df["PTGENDER"]
            .map({"Male": 1, "Female": 0})
            .fillna(0)
            .astype(int)
        )

    if "APOE4" in df.columns:
        df["APOE4"] = (
            pd.to_numeric(df["APOE4"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )

    # =========================
    # Train / test split helper
    # =========================
    def split_data(feature_list):
        features = [f for f in feature_list if f in df.columns]
        if not features:
            raise ValueError(f"No valid features found: {feature_list}")

        X = df[features].copy()
        y = df["target"].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"Split → Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    Xb_train, Xb_test, yb_train, yb_test = split_data(basic_features)
    Xa_train, Xa_test, ya_train, ya_test = split_data(advanced_features)

    # =========================
    # Scaling (tree-safe)
    # =========================
    def scale_and_save(X_train, X_test, name):
        scaler = StandardScaler()
        X_train_s = X_train.copy()
        X_test_s = X_test.copy()

        num_cols = [c for c in X_train.columns if c not in ["PTGENDER", "APOE4"]]

        if num_cols:
            X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test_s[num_cols] = scaler.transform(X_test[num_cols])

        scaler_path = os.path.join(save_dir, f"{name}.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler → {scaler_path}")

        return X_train_s, X_test_s

    Xb_train_s, Xb_test_s = scale_and_save(Xb_train, Xb_test, "scaler_basic")
    Xa_train_s, Xa_test_s = scale_and_save(Xa_train, Xa_test, "scaler_advanced")

    # =========================
    # Save feature lists
    # =========================
    joblib.dump(list(Xb_train_s.columns), os.path.join(save_dir, "basic_features.pkl"))
    joblib.dump(list(Xa_train_s.columns), os.path.join(save_dir, "advanced_features.pkl"))

    print(f"Saved {len(Xb_train_s.columns)} basic features")
    print(f"Saved {len(Xa_train_s.columns)} advanced features")

    # =========================
    # Return
    # =========================
    return (
        Xb_train_s, Xb_test_s, yb_train, yb_test
    ), (
        Xa_train_s, Xa_test_s, ya_train, ya_test
    )
