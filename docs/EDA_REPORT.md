# Exploratory Data Analysis Report — MedIMeta Dermatoscopy (`derm`)

**Generated:** 2026-11-24  
**Script:** `scripts/01_eda.py --config configs/base.yaml`  
**Source Data:** `data/derm/` (MedIMeta standardized HAM10000 / ISIC 2018, DOI: `10.5281/zenodo.7884735`)

---

## 1. Dataset Overview & Integrity Checks

| Property | Value |
| :--- | :--- |
| Dataset ID | `derm` (Dermatoscopy) |
| Task Name | `disease category` (`MULTICLASS_CLASSIFICATION`) |
| Number of Classes | 7 |
| Total Images | 11,720 |
| Predefined Split Sizes | `train`: **10,015** (85.45%) · `val`: **193** (1.65%) · `test`: **1,512** (12.90%) |
| Image Format | `224 × 224 × 3` RGB, `uint8` (standardized from `600 × 450`) |
| Missing Values (`annotations.csv`) | `0` null cells across all 7 columns |
| Unique Source Files | `11,720` / `11,720` (no duplicate filenames) |
| Cross-Split Overlap | `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0` (leak-free) |
| Train RGB Mean (estimated, $n=2000$) | `[0.7607, 0.5412, 0.5641]` |
| Train RGB Std (estimated, $n=2000$) | `[0.1376, 0.1596, 0.1785]` |

---

## 2. Class Distribution & Imbalance Analysis

| ID | Short Code | Official MedIMeta Label | Train | Train % | Val | Val % | Test | Test % |
| :-: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `MEL` | Melanoma | 1,113 | 11.11% | 21 | 10.88% | 171 | 11.31% |
| 1 | `NV` | Melanocytic nevus | 6,705 | **66.95%** | 123 | 63.73% | 909 | 60.12% |
| 2 | `BCC` | Basal cell carcinoma | 514 | 5.13% | 15 | 7.77% | 93 | 6.15% |
| 3 | `AKIEC` | Actinic keratosis / Bowen's disease | 327 | 3.27% | 8 | 4.15% | 43 | 2.84% |
| 4 | `BKL` | Benign keratosis | 1,099 | 10.97% | 22 | 11.40% | 217 | 14.35% |
| 5 | `DF` | Dermatofibroma | **115** | **1.15%** | **1** | **0.52%** | 44 | 2.91% |
| 6 | `VASC` | Vascular lesion | **142** | **1.42%** | **3** | **1.55%** | 35 | 2.31% |
| — | **Total** | — | **10,015** | **100%** | **193** | **100%** | **1,512** | **100%** |

- **Training Imbalance Ratio (Max / Min):** **58.3 : 1** (`NV`: 6,705 vs. `DF`: 115).
- **Inverse-Frequency Class Weights (computed on `train` only):**
  - `NV` receives the smallest weight (`~0.21`), while `DF` (`~12.44`) and `VASC` (`~10.08`) receive the highest weights to prevent majority-class collapse during improved model training.

---

## 3. Methodological Implications for TrustSkin AI

1. **Evaluation Metric Selection:** Because `NV` makes up 60.12% of the test set, raw accuracy can look deceptively high even when rare and malignant classes fail. All experiments will report **Macro F1**, **Balanced Accuracy**, and **per-class precision/recall** alongside Accuracy.
2. **Small Official Validation Set (`val = 193`):**
   - `DF` has only **1** validation image and `VASC` has **3**.
   - **Model Selection:** Using `val_loss` (continuous across all 193 samples) avoids the ±14% step variance of `val_macro_f1`.
   - **Calibration:** Post-hoc **Temperature Scaling** fits only a single scalar parameter $T > 0$ on `val`, making it statistically reliable even with 193 validation samples.
3. **Visual Ambiguity & Domain Characteristics:**
   - Dermoscopic RGB values are strongly red-shifted (`R=0.76`, `G=0.54`, `B=0.56`) with frequent dark circular vignettes, hair artifacts, and subtle pigment-network differences between `MEL`, `NV`, and `BKL`.
   - Rotational/flip invariance holds for dermoscopy (lesions have no canonical orientation), making random horizontal/vertical flips and rotations physically valid augmentations for improved models.

---

## 4. Generated Artifacts
- `outputs/tables/eda_class_distribution.csv`
- `outputs/tables/eda_summary.json`
- `outputs/figures/eda_class_distribution.png` (linear & log-scale split comparison)
- `outputs/figures/eda_sample_grid.png` (7 × 3 representative dermoscopic lesions per class)
