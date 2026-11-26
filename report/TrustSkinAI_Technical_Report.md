# TrustSkin AI: Reliable Recognition of Difficult Skin Lesions

**CS320: Artificial Neural Networks & Deep Learning — BSAI-2k24, Fall 2026**  
National University of Sciences & Technology (NUST), School of Electrical Engineering and Computer Science (SEECS)  
**Project ID:** P08 — Dermatoscopy: Reliable Recognition of Difficult Skin Lesions  

**Group Members:**
1. **Alap Gohar** (502082)
2. **Muhammad Sikandar Hussain** (502808)
3. **Shaheer Muzaffar Khan** (509535)

> **Educational Use Only:** This project is an academic deep-learning research prototype developed for CS320 and is **not** intended for clinical diagnosis, treatment, or medical deployment.

---

## 1. Problem and Motivation

Automated classification of dermoscopic skin lesions is challenging because visually distinct conditions (such as early Melanoma vs. Benign Keratosis or atypical Melanocytic Nevi) share overlapping pigment and border patterns, while real-world dermoscopy datasets exhibit extreme class imbalance.

In safety-sensitive screening settings, **raw accuracy is an insufficient criterion for reliability**. A model can achieve deceptively high accuracy by predicting the majority benign class while failing catastrophically on malignant or rare lesions—and deep neural networks are frequently overconfident when wrong. **TrustSkin AI** addresses CS320 Project **P08** by formulating three explicit, measurable reliability criteria:

1. **Balanced Class-Wise & Difficult-Case Competence:** Dependable recognition across all seven classes and on formally defined difficult subsets (**Balanced Accuracy**, **Macro F1**, per-class recall, and paired **McNemar significance tests**).
2. **Uncertainty & Confidence Calibration:** Predicted probabilities must match empirical accuracy (**Expected Calibration Error (ECE)** and **Brier Score**), and uncertainty scores must separate errors from correct predictions (**Error-Detection AUROC**).
3. **Selective Classification (Abstention / Human Referral):** Uncertain lesions should be referred to a dermatologist, minimizing **Selective Risk** across coverages (**Coverage–Risk curves**, **AURC**, and **malignant error capture rate**).

---

## 2. Dataset and Selected Task

We use the standardized **MedIMeta Dermatoscopy (`derm`)** dataset (Woerner et al., 2025), derived from HAM10000 / ISIC 2018 (Tschandl et al., 2018). The supervised learning task is 7-class mutually exclusive classification (`task_name: "disease category"`) on $224 \times 224 \times 3$ RGB (`uint8`) images.

Exploratory Data Analysis (`scripts/01_eda.py`) confirmed **0 missing values**, **11,720 unique source images**, and **0 cross-split overlap** across the predefined MedIMeta splits (`train`: 10,015; `val`: 193; `test`: 1,512), with training RGB statistics `mean = [0.7607, 0.5412, 0.5641]` and `std = [0.1376, 0.1596, 0.1785]`.

| Class ID | Code | Lesion Category | Train | Train % | Val | Test |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `MEL` | Melanoma *(malignant)* | 1,113 | 11.11% | 21 | 171 |
| 1 | `NV` | Melanocytic nevus | 6,705 | **66.95%** | 123 | 909 |
| 2 | `BCC` | Basal cell carcinoma *(malignant)* | 514 | 5.13% | 15 | 93 |
| 3 | `AKIEC` | Actinic keratosis / Bowen's disease | 327 | 3.27% | 8 | 43 |
| 4 | `BKL` | Benign keratosis | 1,099 | 10.97% | 22 | 217 |
| 5 | `DF` | Dermatofibroma | **115** | **1.15%** | **1** | 44 |
| 6 | `VASC` | Vascular lesion | **142** | **1.42%** | **3** | 35 |
| — | **Total** | — | **10,015** | **100%** | **193** | **1,512** |

*Figure References:* `outputs/figures/eda_class_distribution.png`, `outputs/figures/eda_sample_grid.png`.

### Key Data Insights Driving Methodology
- **58.3 : 1 Class Imbalance:** `NV` is 58.3× larger than `DF` in `train` and represents 60.12% of `test`.
- **Tiny Official Validation Split (`val = 193`):** Because `val` has only **1** `DF` and **3** `VASC` images, `val_macro_f1` fluctuates by ±14 points from a single sample. We therefore use continuous **`val_loss`** for checkpoint selection and single-parameter **Temperature Scaling** for low-variance post-hoc calibration.

---

## 3. Reference Baseline

Following the CS320 common suggested baseline strictly (`configs/baseline_cnn.yaml`), `BaselineCNN` (`src/trustskin/models/baseline_cnn.py`) trains from random initialization with no pretrained weights, no augmentation, and standard Cross-Entropy Loss:
- **Blocks 1–3:** `Conv2d(3→32→64→128, kernel=3, padding=1)` → `ReLU` → `MaxPool2d(2, 2)`
- **Classifier:** `AdaptiveAvgPool2d(1)` (Global Average Pooling) → `Linear(128, 7)`
- **Protocol:** Adam (`lr = 1e-3`, `weight_decay = 0`), batch size `64`, `20` epochs, best checkpoint selected on `val_loss` (epoch 18).

---

## 4. Proposed Approach and Technical Justification

To isolate the contribution of every design decision (CS320 §5), we construct a **single-factor ablation ladder** and compare multiple uncertainty methods on identical MedIMeta splits:

1. **Single-Factor Representation & Imbalance Ladder:**
   - **R1 (`abl_r1_resnet_ce_noaug`):** Pretrained ResNet-18 with a `Dropout(0.3)` + `Linear(512, 7)` head, plain CE, no augmentation — isolates transfer learning alone.
   - **R2 (`abl_r2_resnet_ce_aug`):** R1 + dermoscopy-valid augmentations (flips, ±20° rotation, mild color jitter) — isolates augmentation.
   - **R3 vs. R3s (`resnet18_mc` vs. `abl_r3s_resnet_sqrtw_aug`):** R2 + train-only inverse-frequency class weights with linear (`power = 1.0`) vs. square-root dampening (`power = 0.5`, mean-normalized to 1.0) — tests whether softer weighting preserves rare-class recall while reducing `NV → MEL` false alarms.
   - **Scratch Improved Ablation (`improved_mc_cnn`):** 4-block scratch `UncertaintyCNN` (BatchNorm + Dropout2d + linear weighted CE) documenting the trade-off of harsh class weights on small scratch CNNs.
2. **Uncertainty Estimation Comparison (Member 2):**
   - **Deterministic Softmax** (calibrated single pass) vs. **Monte Carlo Dropout** ($T = 25$ stochastic passes with BatchNorm frozen via `enable_mc_dropout()`) vs. **3-Seed Deep Ensemble** (averaging 3 independently trained, temperature-calibrated R3s seeds: 42, 43, 44).
   - Computes **Predictive Entropy** $H(\bar{p})$ (total uncertainty) and **Mutual Information / BALD** $H(\bar{p}) - \mathbb{E}[H(p^{(t)})]$ (epistemic uncertainty).
3. **Post-Hoc Confidence Calibration (Member 3):**
   - Single-scalar **Temperature Scaling** ($T > 0$) fit via L-BFGS on validation NLL (`val` only), reducing overconfidence without altering top-1 predictions or overfitting `val = 193`.
4. **Selective Classification & Formal Difficult-Case Criteria (Member 3 & P08):**
   - Thresholds are chosen strictly on matched validation confidence scores (`04_calibrate_and_select.py`).
   - Three difficult subsets defined prior to Phase 5: **`hard_6_non_nv`** ($n=603$, all 6 minority classes), **`confusable_triad_mel_nv_bkl`** ($n=1{,}297$, visually overlapping pigmented lesions), and **`baseline_failures`** ($n=513$, exact CS320 baseline errors, analysis only).

---

## 5. Experimental Setup and Reproducibility

- **Strict Split Governance:** `test` ($n=1{,}512$) was never touched during training, early stopping, temperature fitting, or threshold selection.
- **Reproducibility:** Hierarchical YAML configs (`configs/`), deterministic seeding (`seeds 42, 43, 44`), Google Colab Tesla T4 GPU (`PyTorch 2.14.1+cu130`), and **17/17** passing automated tests (`pytest tests/ -v`).
- **Statistical Rigor:** 1,000-resample **bootstrap 95% confidence intervals** (`scripts/07_difficult_case_eval.py`) and exact paired **McNemar tests** comparing every model against `baseline_cnn` on identical test images.

---

## 6. Results and Controlled Comparisons

### 6.1 Controlled Single-Factor Ablation Ladder (Test Split, $n = 1{,}512$, with 95% Bootstrap CIs)

| Step | Experiment | Accuracy [95% CI] | Balanced Acc [95% CI] | Macro F1 [95% CI] | ECE (Before → After) | Brier ↓ | AURC ↓ | Err-AUROC ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | `baseline_cnn` | 66.07% [63.7, 68.4] | 30.85% [27.7, 34.2] | 34.61% [30.5, 38.6] | 0.0546 → 0.0546 | 0.4462 | 0.1334 | 0.8110 |
| **B0-Cal** | `baseline_cnn_calibrated` | 66.07% [63.7, 68.4] | 30.85% [27.7, 34.2] | 34.61% [30.5, 38.6] | 0.0546 → 0.0666 | 0.4485 | 0.1335 | 0.8123 |
| **S1** | `improved_mc_cnn` *(scratch)* | 49.93% [47.6, 52.4] | 44.23% [40.6, 47.7] | 34.28% [31.5, 36.8] | 0.0870 → 0.0770 | 0.5724 | 0.2511 | 0.7875 |
| **R1** | `abl_r1_resnet_ce_noaug` | 76.39% [74.3, 78.6] | 53.77% [49.6, 57.9] | 58.04% [53.4, 62.0] | 0.0952 → 0.0779 | 0.3418 | 0.0747 | 0.8359 |
| **R2** | `abl_r2_resnet_ce_aug` | **79.83%** [77.8, 81.8] | 67.29% [63.4, 71.4] | 67.90% [64.0, 71.4] | 0.0926 → 0.0726 | 0.2955 | 0.0578 | 0.8389 |
| **R3** | `resnet18_mc` *(linear wt)* | 77.25% [75.1, 79.3] | 73.98% [70.0, 77.7] | 70.08% [66.4, 73.4] | 0.0862 → 0.0587 | 0.3218 | 0.0754 | 0.8275 |
| **R3s** | `abl_r3s_resnet_sqrtw_aug` | 78.84% [76.7, 80.9] | 70.37% [66.2, 74.4] | 70.72% [66.6, 74.0] | 0.0693 → 0.0486 | 0.2971 | 0.0626 | 0.8407 |
| **Ens** | **`deep_ensemble` (3× R3s)** | **79.70%** [77.8, 81.8] | **75.70%** [71.7, 79.6] | **74.58%** [71.0, 77.6] | **→ 0.0204** | **0.2815** | **0.0573** | **0.8450** |

*Ablation Takeaways:* Transfer learning alone (`R1`) contributes **+22.9 pts** Balanced Accuracy and **+23.4 pts** Macro F1; dermoscopy augmentation (`R2`) adds **+13.5 pts** Balanced Accuracy and cuts AURC to `0.0578`; square-root class weights (`R3s`) lift Melanoma recall from `40.4% → 70.8%` (+30.4 pts) and lower ECE to `0.0486` without the precision loss of linear weights (`R3`); and ensembling 3 R3s seeds (`Ens`, single-seed mean Macro F1 = `69.57% ± 2.87%`) lifts Macro F1 to **`74.58%`** and cuts ECE to **`0.0204`**!

### 6.2 Uncertainty Method Comparison (`outputs/tables/uncertainty_method_comparison.csv`)

| Method | Accuracy | Balanced Acc | Macro F1 | ECE ↓ | Brier ↓ | AURC ↓ | Err-AUROC (Conf / Ent / MI) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Deterministic Softmax (1 model, TS) | 78.84% | 70.46% | 70.58% | 0.0505 | 0.2980 | 0.0629 | 0.8388 / 0.8393 / 0.5000 |
| MC Dropout (1 model, $T=25$, TS) | 79.17% | 70.98% | 71.04% | 0.0450 | 0.2970 | 0.0620 | 0.8342 / 0.8369 / **0.8043** |
| **Deep Ensemble (3 seeds, TS)** | **79.70%** | **75.70%** | **74.58%** | **0.0204** | **0.2815** | **0.0573** | **0.8450 / 0.8441** / 0.7484 |

### 6.3 Formal Difficult-Case Subsets & Paired McNemar Tests vs. `baseline_cnn`

| Model | `hard_6_non_nv` Acc / AURC | `confusable_triad` Bal Acc | `baseline_failures` Fixed | McNemar `hard_6_non_nv` | McNemar `all_test` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_cnn` | 23.88% / 0.7718 | 49.45% | 0 / 513 (0.0%) | — | — |
| `abl_r1_resnet_ce_noaug` | 49.59% / 0.4278 | 65.51% | 232 / 513 (45.2%) | +155 ($p = 1.1 \times 10^{-26}$) | +156 ($p = 1.6 \times 10^{-19}$) |
| `abl_r2_resnet_ce_aug` | 59.04% / 0.2560 | 66.93% | 281 / 513 (54.8%) | +212 ($p = 2.3 \times 10^{-42}$) | +208 ($p = 6.6 \times 10^{-30}$) |
| `abl_r3s_resnet_sqrtw_aug` | 66.67% / 0.1621 | 71.91% | 321 / 513 (62.6%) | +258 ($p = 4.5 \times 10^{-54}$) | +193 ($p = 3.3 \times 10^{-20}$) |
| **`deep_ensemble`** | **73.30% / 0.1068** | **75.02%** | **341 / 513 (66.5%)** | **+298 ($p = 7.4 \times 10^{-71}$)** | **+206 ($p = 1.3 \times 10^{-21}$)** |

*Figure References:* `outputs/figures/difficult_case_comparison.png`, `outputs/figures/all_models_risk_coverage.png`, `outputs/figures/training_curves.png`.

---

## 7. Error Analysis and Discussion

1. **Majority-Class Collapse in `baseline_cnn`:** `66.07%` accuracy hid `0.0%` recall on `DF`, `21.5%` on `BCC`, and `25.7%` on `MEL` (`200` malignant errors out of `264` malignant cases). Post-hoc Temperature Scaling alone (`baseline_cnn_calibrated`) did not improve ECE (`0.0546 → 0.0666`) or difficult cases.
2. **Controlled Negative / Trade-Off Finding in Scratch `improved_mc_cnn`:** Linear 58× inverse-frequency weights on a low-capacity scratch CNN boosted `hard_6_non_nv` accuracy (`23.9% → 43.8%`, $p = 8.4 \times 10^{-17}$) but caused `244` `NV → MEL` false alarms, significantly regressing `all_test` accuracy (`49.93%`, McNemar -244 net, $p = 1.5 \times 10^{-24}$).
3. **Genuine Difficult-Case & Malignant Safety Gains in `deep_ensemble`:**
   - Total test errors fell from `513 → 307`, and malignant errors (`MEL` + `BCC`) dropped **75%** (`200 → 50`: `89.2%` BCC recall, `76.6%` MEL recall, `77.3%` DF recall).
   - Selective abstention at **70% coverage** ($\tau = 0.6605$) catches **66.0% (`33 / 50`)** of remaining malignant errors, leaving only **17 unreferred malignant errors out of 264 (`93.6%` safe handling)**.
   - Remaining errors concentrate on genuinely ambiguous pigmented lesions (`NV → MEL`: 111, `BKL → MEL`: 34, `MEL → NV`: 23), where conservative over-flagging (`NV → MEL`) strongly outnumbers silent misses (`MEL → NV`).

*Figure References:* `outputs/figures/deep_ensemble_highest_uncertainty_grid.png`, `outputs/figures/deep_ensemble_confident_errors_grid.png`, `outputs/figures/abl_r3s_resnet_sqrtw_aug_reliability_before.png`, `outputs/figures/abl_r3s_resnet_sqrtw_aug_reliability_after.png`.

---

## 8. Limitations

1. **Validation/Test Score-Matching Bug Identified & Fixed:** In our initial Phase 3B run, selecting thresholds on deterministic validation confidences while applying them to MC-averaged test confidences caused a coverage shift (70% target → 56.7% actual). We fixed this in `scripts/04_calibrate_and_select.py` so validation thresholds are computed on matched MC-averaged validation confidences. Still, the tiny official `val` split (`n = 193`, with 1 `DF` and 3 `VASC`) introduces finite-sample variance when estimating quantile thresholds.
2. **Single-Seed Variance:** Across 3 R3s seeds, seed 44 achieved lower accuracy (`71.49%`) than seeds 42 (`78.84%`) and 43 (`77.58%`), demonstrating why multi-seed reporting and ensembling are critical on imbalanced medical datasets.
3. **Residual Confident Errors:** Even with the 3-seed Deep Ensemble at 70% coverage, `17` malignant errors remain unreferred (`deep_ensemble_confident_errors_grid.png`), underscoring that educational ML prototypes must never replace dermatologists.
4. **Ensemble Compute Overhead:** A 3-model Deep Ensemble triples training and inference cost relative to a single deterministic model, though inference remains fast (<1s per lesion on CPU in `app/demo_app.py`).

---

## 9. Conclusion

**TrustSkin AI** demonstrates through a rigorous, single-factor ablation ladder and paired statistical testing that reliable skin-lesion recognition requires combining pretrained representations (`R1`), dermoscopy augmentation (`R2`), dampened imbalance weighting (`R3s`), post-hoc Temperature Scaling, and ensemble uncertainty estimation. Compared to the CS320 baseline, our 3-seed Deep Ensemble increases Balanced Accuracy by **+44.9 points** (`30.85% → 75.70%`), Macro F1 by **+40.0 points** (`34.61% → 74.58%`), `hard_6_non_nv` accuracy by **+49.4 points** (`23.88% → 73.30%`, McNemar $p = 7.4 \times 10^{-71}$), cuts Calibration Error to **`ECE = 0.0204`**, cuts selective risk (AURC) by **57%** (`0.1334 → 0.0573`), and achieves **93.6% malignant-case safety** at 70% coverage—delivered with a live interactive Streamlit demonstration (`app/demo_app.py`).

---

## 10. References, Contributions & AI Disclosure

### References
1. Woerner, S., Jaques, A., & Baumgartner, C. F. (2025). *A comprehensive and easy-to-use multi-domain multi-task medical imaging meta-dataset.* Scientific Data, 12, 666. https://doi.org/10.1038/s41597-025-04866-4
2. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161.
3. Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.* ICML.
4. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles.* NeurIPS.
5. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML.
6. Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for deep neural networks.* NeurIPS.

### Group Contribution Statement
- **Alap Gohar (502082):** MedIMeta dataset inspection & EDA (`scripts/01_eda.py`, `docs/EDA_REPORT.md`), data preprocessing & class-weight pipeline (`src/trustskin/data/datamodule.py`), CS320 reference baseline CNN implementation and evaluation (`src/trustskin/models/baseline_cnn.py`, `scripts/02_train_baseline.py`).
- **Muhammad Sikandar Hussain (502808):** Regularized MC Dropout & ResNet-18 architectures (`src/trustskin/models/uncertainty_cnn.py`), single-factor ablation ladder (`R1 → R2 → R3s`), uncertainty estimation & Deep Ensemble comparison (`src/trustskin/uncertainty/estimator.py`, `scripts/08_ensemble_eval.py`), and Google Colab GPU training runs.
- **Shaheer Muzaffar Khan (509535):** Post-hoc Temperature Scaling & calibration metrics (`src/trustskin/calibration/temperature_scaling.py`), selective classification & matched-score threshold calibration (`src/trustskin/selective/selective_risk.py`, `scripts/04_calibrate_and_select.py`), and formal difficult-case statistical evaluation (`scripts/07_difficult_case_eval.py`).
- **All Members:** Joint comparative analysis, error analysis, interactive Streamlit demonstration (`app/demo_app.py`), technical report, and viva preparation.

### Acknowledgement of External Code & AI Tools
- Open-source libraries used: PyTorch, Torchvision (ResNet-18 ImageNet weights), NumPy, Pandas, SciPy, Scikit-learn, Matplotlib, Seaborn, and Streamlit.
- **AI Tools Usage Disclosure (per CS320 guidelines):** An AI coding assistant (Antigravity) was used to assist with repository scaffolding, modular code organization, debugging Colab dataset staging and the selective-threshold score-matching bug, writing unit tests, and drafting documentation/report structure. All experimental design decisions, dataset verification, Colab GPU training runs, statistical analysis, and final verification were conducted and reviewed by the group members.
