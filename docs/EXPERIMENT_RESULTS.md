# TrustSkin AI — Verified Experimental Results & Difficult-Case Analysis

**Course:** CS320 — Artificial Neural Networks & Deep Learning (BSAI-2k24, Fall 2026)  
**Project ID:** P08 — Reliable Recognition of Difficult Skin Lesions  
**Dataset:** MedIMeta Dermatoscopy (`derm`, task: `"disease category"`, 7 classes, $224 \times 224$ RGB)  
**Evaluation Protocol:** All models trained on the exact `train` split ($n = 10{,}015$), checkpoint-selected and temperature-calibrated strictly on `val` ($n = 193$), and evaluated once on the held-out `test` split ($n = 1{,}512$).  
**Hardware / Environment:** Google Colab Tesla T4 GPU (`PyTorch 2.14.1+cu130`, seed `42`).

---

## 1. Master Controlled Comparison Table (Test Split, $n = 1{,}512$)

| Experiment | Best Epoch | Accuracy | Balanced Acc | Macro Precision | Macro Recall | Macro F1 | ECE (Before → After) | Brier Score ↓ | AURC (Conf / Ent) ↓ | Error-Det AUROC (Ent / MI) ↑ | Learned $T$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_cnn`** *(CS320 3-block scratch)* | 18 | 66.07% | 30.85% | 45.91% | 30.85% | 34.61% | 0.0546 → 0.0546 | 0.4462 | 0.1334 / — | 0.8110 / — | 1.0000 |
| **`baseline_cnn_calibrated`** *(+ Temp Scaling)* | 18 | 66.07% | 30.85% | 45.91% | 30.85% | 34.61% | 0.0546 → 0.0666 | 0.4485 | 0.1335 / 0.1353 | 0.8123 / 0.5000 | 0.9503 |
| **`improved_mc_cnn`** *(4-block scratch + Aug + Weighted CE + MC25 + TS)* | 19 | 49.93% | 44.23% | 33.33% | 44.23% | 34.28% | 0.0870 → 0.0770 | 0.5724 | 0.2511 / 0.2553 | 0.7875 / 0.5846 | 0.9646 |
| **`resnet18_mc`** *(ResNet-18 + Aug + Weighted CE + MC25 + TS)* | 12 | **77.25%** | **73.98%** | **70.88%** | **73.98%** | **70.08%** | **0.0862 → 0.0587** | **0.3218** | **0.0754 / 0.0737** | **0.8275 / 0.8019** | 1.2060 |

---

## 2. Per-Class Classification Performance (Test Split)

| Class (Support) | Baseline CNN (Prec / Rec / F1) | Improved MC-CNN (Prec / Rec / F1) | ResNet18-MC (Prec / Rec / F1) | Key Observation |
| :--- | :---: | :---: | :---: | :--- |
| **0: Melanoma (`MEL`, 171)** *(malignant)* | 57.1% / 25.7% / 35.5% | 26.8% / 66.1% / 38.2% | **47.3% / 71.9% / 57.1%** | Recall jumped **+46.2 pts** over baseline |
| **1: Melanocytic nevus (`NV`, 909)** | 71.5% / 94.1% / 81.2% | 94.1% / 54.0% / 68.6% | **94.5% / 81.2% / 87.3%** | High precision (`94.5%`) without majority-class collapse |
| **2: Basal cell carcinoma (`BCC`, 93)** *(malignant)* | 39.2% / 21.5% / 27.8% | 21.4% / 29.0% / 24.7% | **60.0% / 83.9% / 70.0%** | Recall jumped **+62.4 pts**; F1 jumped **+42.2 pts** |
| **3: Actinic keratosis (`AKIEC`, 43)** | 40.0% / 23.3% / 29.4% | 15.7% / 51.2% / 24.0% | **39.3% / 81.4% / 53.0%** | Strong sensitivity (`81.4%`) on rare precancerous lesions |
| **4: Benign keratosis (`BKL`, 217)** | 40.8% / 28.6% / 33.6% | 41.5% / 35.0% / 38.0% | **72.7% / 65.0% / 68.6%** | F1 more than doubled (`+35.0 pts`) |
| **5: Dermatofibroma (`DF`, 44)** *(rarest)* | 0.0% / 0.0% / 0.0% | 0.0% / 0.0% / 0.0% | **96.7% / 65.9% / 78.4%** | Rescued from complete `0%` collapse to `78.4%` F1 |
| **6: Vascular lesion (`VASC`, 35)** | 72.7% / 22.9% / 34.8% | 33.8% / 74.3% / 46.4% | **85.7% / 68.6% / 76.2%** | F1 jumped **+41.4 pts** |

---

## 3. Confidence Calibration Analysis (Temperature Scaling + MC Dropout)

| Experiment | ECE (Uncalibrated) | ECE (Temp Scaled) | ECE (MC25 + Temp Scaled) | Brier (Uncalibrated → Final) | Learned $T$ (Val) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_cnn_calibrated` | 0.0546 | 0.0666 | 0.0666 *(MC=1)* | 0.4462 → 0.4485 | 0.9503 |
| `improved_mc_cnn` | 0.0870 | 0.0930 | **0.0770** | 0.5781 → 0.5724 | 0.9646 |
| `resnet18_mc` | 0.0862 | 0.0621 | **0.0587** | 0.3318 → **0.3218** | **1.2060** |

### Calibration Takeaways
- **Why `resnet18_mc` benefited strongly:** Higher-capacity models tend to be overconfident (`ECE = 0.0862`). Learning a single scalar temperature $T = 1.2060$ on the `val` split softened overconfident logits (`ECE → 0.0621`), and averaging over $T=25$ MC Dropout passes reduced ECE further to **`0.0587`** (**-32.0% ECE reduction**) and Brier Score to **`0.3218`**.
- **Why `baseline_cnn` did not benefit from Temperature Scaling:** The small 3-block scratch CNN was slightly under-confident on `val` ($T = 0.9503 < 1$), so sharpening logits slightly hurt test ECE (`0.0546 → 0.0666`), showing that post-hoc calibration cannot compensate for an underfitting representation.

---

## 4. Uncertainty Quality & Selective Classification (`resnet18_mc`)

### Uncertainty Error-Detection Quality (AUROC on `resnet18_mc`)
- **Uncalibrated `1 - confidence`:** `0.8189`
- **MC25 Calibrated `1 - confidence`:** `0.8210`
- **MC25 Predictive Entropy:** **`0.8275`** *(best error detector; lowest AURC = `0.0737`)*
- **MC25 Mutual Information (Epistemic / BALD):** **`0.8019`** *(compared to `0.5846` for scratch `improved_mc_cnn`, showing stronger epistemic separation)*

### Selective Classification Operating Points (Thresholds Selected Strictly on `val`)

| Target Coverage (Val) | Actual Test Coverage | Accepted / Referred | Confidence Threshold | Selective Risk (Error Rate) ↓ | Selective Accuracy ↑ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100% | 99.7% | 1,508 / 4 | 0.2861 | 22.61% | 77.39% |
| 90% | 84.7% | 1,280 / 232 | 0.5831 | 16.48% | 83.52% |
| 80% | 71.4% | 1,080 / 432 | 0.7284 | 12.69% | **87.31%** |
| 70% | 56.7% | 858 / 654 | 0.8415 | 8.16% | **91.84%** |
| 60% | 48.7% | 737 / 775 | 0.9119 | 5.43% | **94.57%** |
| 50% | 37.5% | 567 / 945 | 0.9656 | **2.12%** | **97.88%** |

---

## 5. Difficult-Case, Confused-Class & Malignant Safety Analysis

### Malignant Referral Safety (`MEL` + `BCC`, Total = 264 Test Cases)

| Model | Total Malignant Errors (100% Cov) | Caught & Referred @ 90% Cov | Caught & Referred @ 80% Cov | Caught & Referred @ 70% Cov | Unreferred Malignant Errors @ 70% Cov |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_cnn` | 200 | 36 (18.0%) | 74 (37.0%) | 107 (53.5%) | 93 |
| `improved_mc_cnn` | 124 | 34 (27.4%) | 62 (50.0%) | 77 (62.1%) | 47 |
| **`resnet18_mc`** | **63** | 14 (22.2%) | 23 (36.5%) | **31 (49.2%)** | **32** *(87.9% safely classified or referred)* |

### Top Confused Lesion Pairs & Failure Modes
1. **Baseline CNN (`513` total errors):** Overwhelming majority-class bias — top confusions are `BKL → NV` (`125`), `MEL → NV` (`100`), `BCC → NV` (`47`), and `DF → NV` (`34`). Virtually every rare lesion is absorbed into `NV`.
2. **Improved MC-CNN (`757` total errors — Controlled Negative/Trade-Off Finding):**
   - Full linear inverse-frequency class weights (`~58×` weight on `DF` relative to `NV`) combined with early-layer `Dropout2d(0.25)` from random initialization caused severe over-prediction of minority/malignant classes on benign nevi (`NV → MEL`: `244`, `NV → BKL`: `61`, `NV → BCC`: `41`).
   - While Balanced Accuracy rose (`30.85% → 44.23%`) and malignant errors fell (`200 → 124`), precision collapsed on `NV` and `DF` remained at `0%` recall. This demonstrates why raw linear inverse-frequency weights can destabilize small scratch CNNs under extreme 58:1 imbalance.
3. **ResNet18-MC (`344` total errors):**
   - Remaining errors concentrate on genuinely ambiguous pigmented lesions: `NV → MEL` (`101`), `NV → BKL` (`36`), `BKL → MEL` (`29`), and `MEL → NV` (`23`).
   - Notice the clinical asymmetry: `NV → MEL` false alarms (`101`, conservative) vastly outnumber dangerous `MEL → NV` misses (`23`), and `MEL` exhibits the highest mean predictive entropy (`0.6164` nats) of all 7 classes — meaning the model naturally expresses greater uncertainty on the hardest malignant category.

---

## 6. Limitations & Honest Discussion Points (For Technical Report Section 8)

1. **Small Official Validation Split (`val = 193`):** Because `val` contains only `1` Dermatofibroma and `3` Vascular lesion images, validation-selected coverage thresholds exhibit slight coverage shift on `test` (e.g., targeting 70% coverage on `val` yielded `56.7%` actual coverage on `test` for `resnet18_mc`).
2. **Linear Class Weighting Trade-Off:** Linear inverse-frequency weights boosted rare-class recall strongly but induced `NV → MEL` false positives (`101` in `resnet18_mc`, `244` in `improved_mc_cnn`). Dampened weights ($1/\sqrt{n_c}$) or Focal Loss are natural future improvements.
3. **Remaining Confident Errors:** Even at `70%` coverage, `32` malignant errors remain unreferred (`outputs/figures/resnet18_mc_confident_errors_grid.png`), underscoring that educational deep-learning prototypes must never replace clinical dermatological diagnosis.

---

## 7. Report-Ready Figures Index (`outputs/figures/`)

- **Dataset EDA:** `eda_class_distribution.png`, `eda_sample_grid.png`
- **Training Convergence:** `training_curves.png`
- **Multi-Model Selective Risk:** `all_models_risk_coverage.png`
- **Baseline Diagnostics:** `baseline_cnn_confusion_matrix.png`, `baseline_cnn_reliability.png`, `baseline_cnn_highest_uncertainty_grid.png`, `baseline_cnn_confident_errors_grid.png`
- **Improved MC-CNN Diagnostics:** `improved_mc_cnn_confusion_matrix.png`, `improved_mc_cnn_reliability_before.png`, `improved_mc_cnn_reliability_after.png`, `improved_mc_cnn_entropy_correct_vs_error.png`, `improved_mc_cnn_risk_coverage.png`, `improved_mc_cnn_highest_uncertainty_grid.png`, `improved_mc_cnn_confident_errors_grid.png`
- **ResNet18-MC (Best Model) Diagnostics:** `resnet18_mc_confusion_matrix.png`, `resnet18_mc_reliability_before.png`, `resnet18_mc_reliability_after.png`, `resnet18_mc_entropy_correct_vs_error.png`, `resnet18_mc_risk_coverage.png`, `resnet18_mc_highest_uncertainty_grid.png`, `resnet18_mc_confident_errors_grid.png`
