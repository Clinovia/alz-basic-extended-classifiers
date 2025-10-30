import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_adni(csv_path: str, save_dir: str = "models"):
    os.makedirs(save_dir, exist_ok=True)

    # Load CSV safely
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only rows with baseline diagnosis
    df = df.dropna(subset=["DX_bl"])

    # Merge SMC + LMCI → MCI
    df["DX_bl"] = df["DX_bl"].replace({"SMC": "MCI", "LMCI": "MCI"})

    # Encode target
    label_map = {"CN": 0, "MCI": 1, "AD": 2}
    df["target"] = df["DX_bl"].map(label_map)
    df = df.dropna(subset=["target"])

    # ===== Feature Sets =====
    basic_features = [
        "AGE", "MMSE_bl", "CDRSB_bl", "FAQ_bl", "PTEDUCAT",
        "PTGENDER", "APOE4"
    ]
    optional_features = ["RAVLT_immediate_bl", "MOCA_bl", "ADAS13_bl"]

    advanced_features = basic_features + optional_features + [
        "Hippocampus_bl", "Ventricles_bl", "WholeBrain_bl", "Entorhinal_bl",
        "FDG_bl", "AV45_bl", "PIB_bl", "FBB_bl",
        "ABETA_bl", "TAU_bl", "PTAU_bl", "mPACCdigit_bl", "mPACCtrailsB_bl"
    ]

    # Convert numeric columns
    numeric_cols = basic_features + optional_features + [
        "Hippocampus_bl", "Ventricles_bl", "WholeBrain_bl", "Entorhinal_bl",
        "FDG_bl", "AV45_bl", "PIB_bl", "FBB_bl",
        "ABETA_bl", "TAU_bl", "PTAU_bl", "mPACCdigit_bl", "mPACCtrailsB_bl"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numeric missing values with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Encode gender: Male=1, Female=0
    df["PTGENDER"] = df["PTGENDER"].map({"Male": 1, "Female": 0})

    # Fill missing APOE4 with -1
    df["APOE4"] = df["APOE4"].fillna(-1)

    # ===== Split Data =====
    def split_data(features):
        X = df[features].copy()
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    Xb_train, Xb_test, yb_train, yb_test = split_data(basic_features + optional_features)
    Xa_train, Xa_test, ya_train, ya_test = split_data(advanced_features)

    # ===== Scale numeric features =====
    scaler_basic = StandardScaler()
    Xb_train_scaled = Xb_train.copy()
    Xb_test_scaled = Xb_test.copy()
    num_basic_cols = [c for c in Xb_train.columns if c not in ["PTGENDER", "APOE4"]]
    Xb_train_scaled[num_basic_cols] = scaler_basic.fit_transform(Xb_train[num_basic_cols])
    Xb_test_scaled[num_basic_cols] = scaler_basic.transform(Xb_test[num_basic_cols])
    joblib.dump(scaler_basic, os.path.join(save_dir, "scaler_basic.pkl"))

    scaler_adv = StandardScaler()
    Xa_train_scaled = Xa_train.copy()
    Xa_test_scaled = Xa_test.copy()
    num_adv_cols = [c for c in Xa_train.columns if c not in ["PTGENDER", "APOE4"]]
    Xa_train_scaled[num_adv_cols] = scaler_adv.fit_transform(Xa_train[num_adv_cols])
    Xa_test_scaled[num_adv_cols] = scaler_adv.transform(Xa_test[num_adv_cols])
    joblib.dump(scaler_adv, os.path.join(save_dir, "scaler_advanced.pkl"))

    # ===== Save dataset sizes =====
    sizes = {
        "basic_train": len(Xb_train),
        "basic_test": len(Xb_test),
        "advanced_train": len(Xa_train),
        "advanced_test": len(Xa_test)
    }
    print("Dataset sizes:", sizes)
    joblib.dump(sizes, os.path.join(save_dir, "dataset_sizes.pkl"))

    return (Xb_train_scaled, Xb_test_scaled, yb_train, yb_test), \
           (Xa_train_scaled, Xa_test_scaled, ya_train, ya_test)
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_adni(csv_path: str):
    df = pd.read_csv(csv_path)

    # Drop rows without baseline diagnosis
    df = df.dropna(subset=["DX_bl"])

    # Merge classes: SMC + LMCI → MCI
    df["DX_bl"] = df["DX_bl"].replace({"SMC": "MCI", "LMCI": "MCI"})

    # Simplify gender
    df["PTGENDER"] = df["PTGENDER"].map({"Male": 1, "Female": 0})

    # Fill missing APOE4 with -1 (unknown)
    df["APOE4"] = df["APOE4"].fillna(-1)

    # ===== BASIC FEATURES =====
    basic_features = [
        "AGE", "MMSE", "CDRSB", "FAQ", "PTEDUCAT",
        "PTGENDER", "APOE4"
    ]
    
    # Optional but helpful neuropsych measures
    optional_features = [
        "RAVLT_immediate", "MOCA", "ADAS13"
    ]

    # ===== ADVANCED FEATURES =====
    advanced_features = basic_features + optional_features + [
        "Hippocampus", "Ventricles", "WholeBrain", "Entorhinal",
        "FDG", "AV45", "PIB", "FBB",
        "ABETA", "TAU", "PTAU", "mPACCdigit", "mPACCtrailsB"
    ]

    df_basic = df[basic_features + ["DX_bl"]].copy()
    df_advanced = df[advanced_features + ["DX_bl"]].copy()

    # Handle missing values
    df_basic = df_basic.fillna(df_basic.median(numeric_only=True))
    df_advanced = df_advanced.fillna(df_advanced.median(numeric_only=True))

    # Encode target variable
    label_map = {"CN": 0, "MCI": 1, "AD": 2}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_adni(csv_path: str, save_dir: str = "models"):
    os.makedirs(save_dir, exist_ok=True)

    # Load CSV safely
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only rows with baseline diagnosis
    df = df.dropna(subset=["DX_bl"])

    # Merge SMC + LMCI → MCI
    df["DX_bl"] = df["DX_bl"].replace({"SMC": "MCI", "LMCI": "MCI"})

    # Encode target
    label_map = {"CN": 0, "MCI": 1, "AD": 2}
    df["target"] = df["DX_bl"].map(label_map)
    df = df.dropna(subset=["target"])

    # ===== Feature Sets =====
    basic_features = [
        "AGE", "MMSE_bl", "CDRSB_bl", "FAQ_bl", "PTEDUCAT",
        "PTGENDER", "APOE4"
    ]
    optional_features = ["RAVLT_immediate_bl", "MOCA_bl", "ADAS13_bl"]

    advanced_features = basic_features + optional_features + [
        "Hippocampus_bl", "Ventricles_bl", "WholeBrain_bl", "Entorhinal_bl",
        "FDG_bl", "AV45_bl", "PIB_bl", "FBB_bl",
        "ABETA_bl", "TAU_bl", "PTAU_bl", "mPACCdigit_bl", "mPACCtrailsB_bl"
    ]

    # Convert numeric columns
    numeric_cols = basic_features + optional_features + [
        "Hippocampus_bl", "Ventricles_bl", "WholeBrain_bl", "Entorhinal_bl",
        "FDG_bl", "AV45_bl", "PIB_bl", "FBB_bl",
        "ABETA_bl", "TAU_bl", "PTAU_bl", "mPACCdigit_bl", "mPACCtrailsB_bl"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numeric missing values with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Encode gender: Male=1, Female=0
    df["PTGENDER"] = df["PTGENDER"].map({"Male": 1, "Female": 0})

    # Fill missing APOE4 with -1
    df["APOE4"] = df["APOE4"].fillna(-1)

    # ===== Split Data =====
    def split_data(features):
        X = df[features].copy()
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    Xb_train, Xb_test, yb_train, yb_test = split_data(basic_features + optional_features)
    Xa_train, Xa_test, ya_train, ya_test = split_data(advanced_features)

    # ===== Scale numeric features =====
    scaler_basic = StandardScaler()
    Xb_train_scaled = Xb_train.copy()
    Xb_test_scaled = Xb_test.copy()
    num_basic_cols = [c for c in Xb_train.columns if c not in ["PTGENDER", "APOE4"]]
    Xb_train_scaled[num_basic_cols] = scaler_basic.fit_transform(Xb_train[num_basic_cols])
    Xb_test_scaled[num_basic_cols] = scaler_basic.transform(Xb_test[num_basic_cols])
    joblib.dump(scaler_basic, os.path.join(save_dir, "scaler_basic.pkl"))

    scaler_adv = StandardScaler()
    Xa_train_scaled = Xa_train.copy()
    Xa_test_scaled = Xa_test.copy()
    num_adv_cols = [c for c in Xa_train.columns if c not in ["PTGENDER", "APOE4"]]
    Xa_train_scaled[num_adv_cols] = scaler_adv.fit_transform(Xa_train[num_adv_cols])
    Xa_test_scaled[num_adv_cols] = scaler_adv.transform(Xa_test[num_adv_cols])
    joblib.dump(scaler_adv, os.path.join(save_dir, "scaler_advanced.pkl"))

    # ===== Save dataset sizes =====
    sizes = {
        "basic_train": len(Xb_train),
        "basic_test": len(Xb_test),
        "advanced_train": len(Xa_train),
        "advanced_test": len(Xa_test)
    }
    print("Dataset sizes:", sizes)
    joblib.dump(sizes, os.path.join(save_dir, "dataset_sizes.pkl"))

    return (Xb_train_scaled, Xb_test_scaled, yb_train, yb_test), \
           (Xa_train_scaled, Xa_test_scaled, ya_train, ya_test)
    df_basic["target"] = df_basic["DX_bl"].map(label_map)
    df_advanced["target"] = df_advanced["DX_bl"].map(label_map)

    # Train-test split
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        df_basic.drop(columns=["DX_bl", "target"]),
        df_basic["target"], test_size=0.2, random_state=42, stratify=df_basic["target"]
    )

    Xa_train, Xa_test, ya_train, ya_test = train_test_split(
        df_advanced.drop(columns=["DX_bl", "target"]),
        df_advanced["target"], test_size=0.2, random_state=42, stratify=df_advanced["target"]
    )

    return (Xb_train, Xb_test, yb_train, yb_test), (Xa_train, Xa_test, ya_train, ya_test)

