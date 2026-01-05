# Alzheimer’s Disease Classifier (Baseline Visits Only)

This project implements machine learning classifiers for predicting cognitive status — **Cognitively Normal (CN)**, **Mild Cognitive Impairment (MCI)**, and **Alzheimer’s Disease (AD)** — using ADNI baseline visit data (VISCODE='bl').

Two models are trained:

1. **Basic Model**: Uses demographic and standard neuropsychological measures.
2. **Advanced Model**: Adds brain imaging and biomarker features to the basic set.

---

## Dataset

- Total records loaded: 16,421  
- Baseline (VISCODE='bl') records kept: 2,430  
- Unique patients after removing duplicates by RID: 2,430  

**Dataset split:**

| Model Type | Train | Test |
|------------|-------|------|
| Basic      | 1,596 | 400  |
| Advanced   | 1,596 | 400  |

---

## Features

**Basic features:**

- AGE, MMSE_bl, CDRSB_bl, FAQ_bl, PTEDUCAT, PTGENDER, APOE4  
- Optional neuropsych measures: RAVLT_immediate_bl, MOCA_bl, ADAS13_bl  

**Advanced features:**

- All basic features + optional neuropsych measures  
- Brain imaging & biomarkers: Hippocampus_bl, Ventricles_bl, WholeBrain_bl, Entorhinal_bl, FDG_bl, AV45_bl, PIB_bl, FBB_bl, ABETA_bl, TAU_bl, PTAU_bl, mPACCdigit_bl, mPACCtrailsB_bl  

---

## Model Performance (Test Set)

### Basic Model

| Class | Precision | Recall / Sensitivity | F1-score | Specificity |
|-------|-----------|--------------------|----------|------------|
| CN    | 0.70      | 0.70               | 0.70     | 0.88       |
| MCI   | 0.78      | 0.81               | 0.79     | 0.72       |
| AD    | 0.92      | 0.80               | 0.86     | 0.96       |

**Overall Accuracy:** 0.78  

### Advanced Model

| Class | Precision | Recall / Sensitivity | F1-score | Specificity |
|-------|-----------|--------------------|----------|------------|
| CN    | 0.69      | 0.73               | 0.71     | 0.87       |
| MCI   | 0.79      | 0.79               | 0.79     | 0.73       |
| AD    | 0.91      | 0.82               | 0.86     | 0.96       |

**Overall Accuracy:** 0.78  

> Note: Sensitivity is equivalent to recall. Specificity measures how well the model identifies patients without the condition.

---

## Usage

1. Clone repository:

```bash
git clone <repo-url>
cd alz-basic-advanced-classifiers
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Train the model:
```bash
python scripts/train.py
```

Notes
Baseline only: Only first visits (VISCODE='bl') are used to ensure unique patients in train/test split.
Clinical relevance: Sensitivity and specificity are included for clinician interpretation.
Future steps: Models can be extended to include follow-up visits, but patient grouping should be maintained to avoid data leakage.