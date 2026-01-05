import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_baseline_dataset(csv_path: str = "data/adni_merge_baseline.csv"):
    """
    Comprehensive analysis of ADNI baseline dataset.
    Analyzes class distribution, missing data, and feature availability.
    """
    
    # Load data
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total samples loaded: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Drop samples with missing DX
    if "DX" in df.columns:
        initial_count = len(df)
        df = df.dropna(subset=["DX"])
        dropped_count = initial_count - len(df)
        print(f"Dropped {dropped_count} samples with missing DX")
        print(f"Remaining samples: {len(df)}\n")
        
    # ========================================
    # 1. CLASS DISTRIBUTION
    # ========================================
    print("=" * 80)
    print("CLASS DISTRIBUTION (Number of Samples per Class)")
    print("=" * 80)
    
    if "DX" in df.columns:
        # Raw distribution
        class_counts = df["DX"].value_counts()
        class_pct = df["DX"].value_counts(normalize=True) * 100
        
        print("\nRaw Counts:")
        for dx, count in class_counts.items():
            pct = class_pct[dx]
            print(f"  {dx:10s}: {count:5d} ({pct:5.1f}%)")
        
        # After merging SMC + EMCI + LMCI → MCI
        df_merged = df.copy()
        df_merged["DX"] = df_merged["DX"].replace({"SMC": "MCI", "EMCI": "MCI", "LMCI": "MCI"})
        
        print("\nAfter merging SMC + EMCI + LMCI → MCI:")
        merged_counts = df_merged["DX"].value_counts()
        merged_pct = df_merged["DX"].value_counts(normalize=True) * 100
        
        for dx, count in merged_counts.items():
            pct = merged_pct[dx]
            print(f"  {dx:10s}: {count:5d} ({pct:5.1f}%)")
        
        # Check for missing diagnosis
        missing_dx = df["DX"].isna().sum()
        print(f"\nMissing DX: {missing_dx}")
        
    else:
        print("ERROR: DX column not found!")
    
    print()
    
    # ========================================
    # 2. FEATURE CATEGORIES
    # ========================================
    print("=" * 80)
    print("FEATURE AVAILABILITY & MISSING DATA")
    print("=" * 80)
    
    # Define feature categories
    demographics = ["AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY"]
    genetic = ["APOE4"]
    cognitive_tests = [
        "MMSE", "MOCA", "ADAS11", "ADAS13", "ADASQ4",
        "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", 
        "RAVLT_perc_forgetting", "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ"
    ]
    functional_assessments = [
        "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", 
        "EcogPtOrgan", "EcogPtDivatt", "EcogPtTotal",
        "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", 
        "EcogSPOrgan", "EcogSPDivatt", "EcogSPTotal"
    ]
    imaging = [
        "Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", 
        "Fusiform", "MidTemp", "ICV"
    ]
    pet_biomarkers = ["FDG", "PIB", "AV45", "FBB"]
    csf_biomarkers = ["ABETA", "TAU", "PTAU"]
    other_cognitive = ["mPACCdigit", "mPACCtrailsB"]
    
    # Circular/excluded features
    excluded = ["CDRSB", "DX"]
    
    feature_groups = {
        "Demographics": demographics,
        "Genetic": genetic,
        "Cognitive Tests": cognitive_tests,
        "Functional Assessments": functional_assessments,
        "Imaging (MRI)": imaging,
        "PET Biomarkers": pet_biomarkers,
        "CSF Biomarkers": csf_biomarkers,
        "Other Cognitive": other_cognitive,
        "⚠️  EXCLUDED (Circular)": excluded
    }
    
    # Analyze each group
    for group_name, features in feature_groups.items():
        print(f"\n{group_name}:")
        print("-" * 80)
        
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if available_features:
            for feat in available_features:
                total = len(df)
                non_null = df[feat].notna().sum()
                null_count = df[feat].isna().sum()
                null_pct = (null_count / total) * 100
                
                # Status indicator
                if null_pct < 10:
                    status = "✅"
                elif null_pct < 50:
                    status = "⚠️ "
                else:
                    status = "❌"
                
                print(f"  {status} {feat:25s}: {non_null:5d}/{total} available ({null_pct:5.1f}% missing)")
        
        if missing_features:
            print(f"  ❌ Not in dataset: {', '.join(missing_features)}")
    
    print()
    
    # ========================================
    # 3. RECOMMENDED FEATURE SETS
    # ========================================
    print("=" * 80)
    print("RECOMMENDED FEATURE SETS")
    print("=" * 80)
    
    # Merge all MCI subtypes for feature set analysis
    df_analysis = df.copy()
    df_analysis["DX"] = df_analysis["DX"].replace({"SMC": "MCI", "LMCI": "MCI", "EMCI": "MCI"})
    
    # Basic feature set (most complete)
    basic_features = [
        "AGE", "PTGENDER", "PTEDUCAT", "APOE4", 
        "MMSE", "FAQ", "RAVLT_immediate", "ADAS13"
    ]
    
    # Extended feature set (adds more cognitive + imaging)
    extended_features = basic_features + [
        "ADAS11", "RAVLT_learning", "RAVLT_forgetting", 
        "TRABSCOR", "Ventricles", "WholeBrain", "ICV"
    ]
    
    # Extended + Hippocampus (key biomarker but more missing data)
    extended_with_hippo = extended_features + ["Hippocampus"]
    
    def count_complete_cases(features, show_class_dist=True):
        """Count samples with all features non-null"""
        existing_features = [f for f in features if f in df_analysis.columns]
        if not existing_features:
            return 0, [], None
        
        # Filter for complete cases
        df_complete = df_analysis[existing_features + ["DX"]].dropna()
        complete_count = len(df_complete)
        
        # Get class distribution if requested
        class_dist = None
        if show_class_dist and "DX" in df_complete.columns:
            class_dist = df_complete["DX"].value_counts().to_dict()
        
        return complete_count, existing_features, class_dist
    
    print("\n1. BASIC MODEL (Screening/Primary Care)")
    basic_complete, basic_existing, basic_dist = count_complete_cases(basic_features)
    print(f"   Features ({len(basic_existing)}): {', '.join(basic_existing)}")
    print(f"   Complete cases: {basic_complete}/{len(df_analysis)} ({basic_complete/len(df_analysis)*100:.1f}%)")
    if basic_dist:
        print(f"   Class distribution:")
        for dx, count in sorted(basic_dist.items()):
            print(f"     {dx}: {count} ({count/basic_complete*100:.1f}%)")
    print(f"   ✅ Recommended for general screening")
    
    print("\n2. EXTENDED MODEL (Specialty Clinic)")
    ext_complete, ext_existing, ext_dist = count_complete_cases(extended_features)
    print(f"   Features ({len(ext_existing)}): {', '.join(ext_existing)}")
    print(f"   Complete cases: {ext_complete}/{len(df_analysis)} ({ext_complete/len(df_analysis)*100:.1f}%)")
    if ext_dist:
        print(f"   Class distribution:")
        for dx, count in sorted(ext_dist.items()):
            print(f"     {dx}: {count} ({count/ext_complete*100:.1f}%)")
    print(f"   ✅ Adds cognitive tests + MRI imaging")
    
    print("\n3. EXTENDED + HIPPOCAMPUS (Key Biomarker)")
    hippo_complete, hippo_existing, hippo_dist = count_complete_cases(extended_with_hippo)
    print(f"   Features ({len(hippo_existing)}): {', '.join(hippo_existing)}")
    print(f"   Complete cases: {hippo_complete}/{len(df_analysis)} ({hippo_complete/len(df_analysis)*100:.1f}%)")
    if hippo_dist:
        print(f"   Class distribution:")
        for dx, count in sorted(hippo_dist.items()):
            print(f"     {dx}: {count} ({count/hippo_complete*100:.1f}%)")
    print(f"   ⚠️  More missing data but includes critical AD biomarker")
    
    print()
    
    # ========================================
    # 4. DATA QUALITY SUMMARY
    # ========================================
    print("=" * 80)
    print("DATA QUALITY SUMMARY")
    print("=" * 80)
    
    # Features with <10% missing (good quality)
    good_features = []
    moderate_features = []
    poor_features = []
    
    for col in df.columns:
        if col in ["RID", "PTID", "VISCODE", "EXAMDATE", "DX"]:
            continue  # Skip metadata
        
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        
        if missing_pct < 10:
            good_features.append((col, missing_pct))
        elif missing_pct < 50:
            moderate_features.append((col, missing_pct))
        else:
            poor_features.append((col, missing_pct))
    
    print(f"\n✅ High Quality (<10% missing): {len(good_features)} features")
    if len(good_features) <= 20:
        for feat, pct in sorted(good_features, key=lambda x: x[1]):
            print(f"   {feat}: {pct:.1f}% missing")
    else:
        print(f"   Top 10 most complete:")
        for feat, pct in sorted(good_features, key=lambda x: x[1])[:10]:
            print(f"   {feat}: {pct:.1f}% missing")
    
    print(f"\n⚠️  Moderate Quality (10-50% missing): {len(moderate_features)} features")
    print(f"\n❌ Poor Quality (>50% missing): {len(poor_features)} features")
    
    # ========================================
    # 5. SAMPLE SIZE RECOMMENDATIONS
    # ========================================
    print()
    print("=" * 80)
    print("SAMPLE SIZE ANALYSIS")
    print("=" * 80)
    
    if "DX" in df.columns:
        df_clean = df.copy()
        df_clean["DX"] = df_clean["DX"].replace({"SMC": "MCI", "LMCI": "MCI"})
        df_clean = df_clean.dropna(subset=["DX"])
        
        print(f"\nTotal samples after cleaning: {len(df_clean)}")
        print(f"\nClass distribution (for train/test split planning):")
        
        for dx_class in ["CN", "MCI", "AD"]:
            if dx_class in df_clean["DX"].values:
                class_count = (df_clean["DX"] == dx_class).sum()
                
                # Estimate train/test split (80/20)
                train_count = int(class_count * 0.8)
                test_count = class_count - train_count
                
                print(f"\n  {dx_class}:")
                print(f"    Total: {class_count}")
                print(f"    Train (80%): ~{train_count}")
                print(f"    Test (20%): ~{test_count}")
                
                if test_count < 50:
                    print(f"    ⚠️  Warning: Small test set for this class")
        
        print(f"\nRecommendations:")
        print(f"  • Use stratified K-fold cross-validation (5 or 10 folds)")
        print(f"  • Consider class balancing if any class < 500 samples")
        print(f"  • Start with basic model to maximize sample usage")
        print(f"  • Build separate advanced model on complete-case subset")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_baseline_dataset("data/adni_merge_baseline.csv")