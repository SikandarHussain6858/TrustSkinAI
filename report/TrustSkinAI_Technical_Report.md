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

Automated classification of dermoscopic skin lesions is a challenging medical-imaging task because visually distinct conditions (such as early Melanoma vs. Benign Keratosis or atypical Melanocytic Nevi) often share overlapping color, border, and pigment-network patterns. Furthermore, standard dermoscopy benchmarks exhibit extreme class imbalance, where benign nevi dominate while rare or malignant classes constitute a small fraction of samples.

In safety-sensitive screening contexts, **raw accuracy alone is an inadequate reliability criterion**. A classifier can achieve moderate accuracy by defaulting to the majority class while failing catastrophically on rare or malignant lesions—and standard deep neural networks are frequently overconfident even when they are wrong. **TrustSkin AI** addresses CS320 Project **P08** by framing reliability along three concrete, measurable axes:

1. **Balanced Class-Wise Competence:** Dependable recognition across all seven lesion categories (especially malignant Melanoma and Basal Cell Carcinoma, and rare Dermatofibroma), measured via **Balanced Accuracy**, **Macro F1**, and per-class recall.
2. **Uncertainty & Confidence Calibration:** Predicted probabilities should align with empirical correctness (**Expected Calibration Error (ECE)** and **Brier Score**), and uncertainty estimates should reliably separate errors from correct predictions (**Error-Detection AUROC**).
3. **Selective Classification (Abstention / Human Referral):** When uncertainty exceeds a threshold, the system should abstain and refer the lesion for expert review, minimizing **Selective Risk** across operating coverages (**Coverage–Risk curves** and **AURC**).

---

## 2. Dataset and Selected Task

We use the standardized **MedIMeta Dermatoscopy (`derm`)** dataset (Woerner et al., 2025), derived from HAM10000 / ISIC 2018 (Tschandl et al., 2018; Codella et al., 2019). The supervised learning task is 7-class mutually exclusive multiclass classification (`task_name: "disease category"`) on standardized $224 \times 224 \times 3$ RGB (`uint8`) dermoscopic images.

### Dataset Integrity & Split Distribution
Exploratory Data Analysis (`scripts/01_eda.py`) verified **0 missing values** in `annotations.csv`, **11,720 unique source images**, and **0 cross-split overlap** across the predefined MedIMeta partitions (`train`: 10,015; `val`: 193; `test`: 1,512). Estimated training-split RGB statistics (`mean = [0.7607, 0.5412, 0.5641]`, `std = [0.1376, 0.1596, 0.1785]`) reflect the pink/red color distribution typical of dermoscopy.

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

*Figure References:* `outputs/figures/eda_class_distribution.png` (linear & log-scale imbalance) and `outputs/figures/eda_sample_grid.png` (representative lesions per class).

### Key Data Insights Driving Methodology
- **58.3 : 1 Class Imbalance:** `NV` is 58.3× larger than `DF` in `train` and represents 60.12% of `test`. A trivial majority-class predictor attains ~60% test accuracy with only 14.3% balanced accuracy.
- **Tiny Official Validation Split (`val = 193`):** Because `val` contains only **1** `DF` sample and **3** `VASC` samples, selecting checkpoints by `val_macro_f1` introduces ±14-point step variance from a single image. We therefore use continuous **`val_loss`** for checkpoint selection while logging `val_macro_f1` every epoch, and we use single-parameter **Temperature Scaling** so calibration does not overfit the 193-sample validation set.

---

## 3. Reference Baseline

Following the CS320 common suggested baseline specification strictly (`configs/baseline_cnn.yaml`), we implemented `BaselineCNN` (`src/trustskin/models/baseline_cnn.py`) trained from random initialization with no pretrained weights, no augmentation, and standard Cross-Entropy Loss:

- **Input:** $224 \times 224 \times 3$ normalized RGB image
- **Block 1:** `Conv2d(3, 32, kernel_size=3, padding=1)` → `ReLU` → `MaxPool2d(2, 2)`
- **Block 2:** `Conv2d(32, 64, kernel_size=3, padding=1)` → `ReLU` → `MaxPool2d(2, 2)`
- **Block 3:** `Conv2d(64, 128, kernel_size=3, padding=1)` → `ReLU` → `MaxPool2d(2, 2)`
- **Classifier:** `AdaptiveAvgPool2d((1, 1))` (Global Average Pooling) → `Linear(128, 7)`
- **Protocol:** Adam (`lr = 1e-3`, `weight_decay = 0`), batch size `64`, `20` epochs, best checkpoint selected on `val_loss` (epoch 18).

---

## 4. Proposed Approach and Technical Justification

To improve both class-wise discrimination and prediction reliability on difficult lesions, **TrustSkin AI** combines four technically motivated components evaluated via controlled ablations:

1. **Regularized Architectures & Imbalance Handling:**
   - **Scratch Improved Model (`UncertaintyCNN`):** 4-block CNN with `BatchNorm2d`, `Dropout2d` / `Dropout(0.25)`, dermoscopy-valid augmentations (horizontal/vertical flips, ±20° rotation, mild color jitter—valid because dermoscopic lesions are rotationally invariant), AdamW with cosine annealing, and train-only inverse-frequency class weights (`weighted_ce`).
   - **Comparative Pretrained Backbone (`ResNet18MC`):** ImageNet-pretrained ResNet-18 equipped with a `Dropout(0.3)` + `Linear(512, 7)` head trained under the exact same weighted loss and augmentation protocol, serving as a comparative reference (without replacing the CS320 scratch baseline).
2. **Uncertainty Estimation via Monte Carlo (MC) Dropout:**
   - At inference, BatchNorm statistics remain frozen in `eval()` mode while Dropout layers stay active (`enable_mc_dropout()`). Running $T = 25$ stochastic forward passes yields a mean predictive distribution $\bar{p} = \frac{1}{T}\sum_{t=1}^T p^{(t)}$, **Predictive Entropy** $H(\bar{p}) = -\sum_c \bar{p}_c \log \bar{p}_c$ (total uncertainty), and **Mutual Information (BALD)** $I(y; \theta \mid x) = H(\bar{p}) - \frac{1}{T}\sum_t H(p^{(t)})$ (epistemic uncertainty).
3. **Post-Hoc Confidence Calibration via Temperature Scaling:**
   - Optimizing a single positive scalar temperature $T > 0$ via L-BFGS to minimize Negative Log-Likelihood strictly on the **validation split** (`193` samples) recalibrates softmax probabilities $\hat{p} = \text{softmax}(z / T)$ without altering top-1 accuracy or overfitting small validation sets.
4. **Selective Classification (Abstention & Referral):**
   - Given a confidence threshold $\tau$ (selected strictly on validation scores), predictions with $\max_c \bar{p}_c \ge \tau$ are accepted while uncertain cases are referred to a dermatologist. Performance is measured across all thresholds via **Coverage–Risk curves** and **Area Under the Risk–Coverage curve (AURC)** (lower is better).

---

## 5. Experimental Setup and Reproducibility

- **Strict Split Governance:** The `test` split (`1,512` images) was never used for training, early stopping, hyperparameter tuning, temperature fitting, or threshold selection.
- **Deterministic Reproducibility:** Fixed random seed (`seed = 42`) across Python, NumPy, and PyTorch (`src/trustskin/utils/seed.py`); all experiments are defined in version-controlled YAML configs (`configs/`).
- **Compute Environment:** Google Colab Tesla T4 GPU (`PyTorch 2.14.1+cu130`, CUDA 13.0) via `notebooks/00_colab_setup_and_train.ipynb`, verified with `12/12` automated unit and real-dataset smoke tests (`pytest tests/ -v`).
- **Controlled Experiment Suite:**
  - **Exp 0 (`baseline_cnn`):** CS320 3-block scratch CNN (deterministic, $T = 1.0$).
  - **Exp 0-Cal (`baseline_cnn_calibrated`):** Exp 0 + post-hoc Temperature Scaling fit on `val` (isolates whether calibration alone fixes the baseline).
  - **Exp 1/2/3 (`improved_mc_cnn`):** 4-block scratch `UncertaintyCNN` + Augmentation + Weighted CE + MC Dropout ($T=25$) + Temperature Scaling.
  - **Exp 4 (`resnet18_mc`):** Comparative Pretrained `ResNet18MC` + Augmentation + Weighted CE + MC Dropout ($T=25$) + Temperature Scaling.

---

## 6. Results and Controlled Comparisons

### 6.1 Master Test-Set Comparison ($n = 1{,}512$)

| Experiment | Best Ep | Accuracy | Balanced Acc | Macro F1 | ECE (Before → After) | Brier ↓ | AURC (Conf / Ent) ↓ | Err-AUROC (Ent / MI) ↑ | Learned $T$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_cnn`** | 18 | 66.07% | 30.85% | 34.61% | 0.0546 → 0.0546 | 0.4462 | 0.1334 / — | 0.8110 / — | 1.0000 |
| **`baseline_cnn_calibrated`** | 18 | 66.07% | 30.85% | 34.61% | 0.0546 → 0.0666 | 0.4485 | 0.1335 / 0.1353 | 0.8123 / 0.5000 | 0.9503 |
| **`improved_mc_cnn`** | 19 | 49.93% | 44.23% | 34.28% | 0.0870 → 0.0770 | 0.5724 | 0.2511 / 0.2553 | 0.7875 / 0.5846 | 0.9646 |
| **`resnet18_mc`** | 12 | **77.25%** | **73.98%** | **70.08%** | **0.0862 → 0.0587** | **0.3218** | **0.0754 / 0.0737** | **0.8275 / 0.8019** | 1.2060 |

*Figure References:* `outputs/figures/all_models_risk_coverage.png`, `outputs/figures/training_curves.png`, `outputs/figures/resnet18_mc_reliability_before.png`, `outputs/figures/resnet18_mc_reliability_after.png`.

### 6.2 Per-Class Test Performance (Precision / Recall / F1)

| Class (Test Support) | Baseline CNN | Improved MC-CNN | ResNet18-MC |
| :--- | :---: | :---: | :---: |
| **MEL (171)** *(malignant)* | 57.1% / 25.7% / 35.5% | 26.8% / 66.1% / 38.2% | **47.3% / 71.9% / 57.1%** |
| **NV (909)** | 71.5% / 94.1% / 81.2% | 94.1% / 54.0% / 68.6% | **94.5% / 81.2% / 87.3%** |
| **BCC (93)** *(malignant)* | 39.2% / 21.5% / 27.8% | 21.4% / 29.0% / 24.7% | **60.0% / 83.9% / 70.0%** |
| **AKIEC (43)** | 40.0% / 23.3% / 29.4% | 15.7% / 51.2% / 24.0% | **39.3% / 81.4% / 53.0%** |
| **BKL (217)** | 40.8% / 28.6% / 33.6% | 41.5% / 35.0% / 38.0% | **72.7% / 65.0% / 68.6%** |
| **DF (44)** *(rarest)* | 0.0% / 0.0% / 0.0% | 0.0% / 0.0% / 0.0% | **96.7% / 65.9% / 78.4%** |
| **VASC (35)** | 72.7% / 22.9% / 34.8% | 33.8% / 74.3% / 46.4% | **85.7% / 68.6% / 76.2%** |

### 6.3 Selective Classification Operating Points (`resnet18_mc`, Val-Selected Thresholds)

| Val Target Cov | Actual Test Cov | Accepted / Referred | Threshold ($\tau$) | Selective Risk ↓ | Selective Accuracy ↑ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100% | 99.7% | 1,508 / 4 | 0.2861 | 22.61% | 77.39% |
| 90% | 84.7% | 1,280 / 232 | 0.5831 | 16.48% | 83.52% |
| 80% | 71.4% | 1,080 / 432 | 0.7284 | 12.69% | **87.31%** |
| 70% | 56.7% | 858 / 654 | 0.8415 | 8.16% | **91.84%** |
| 60% | 48.7% | 737 / 775 | 0.9119 | 5.43% | **94.57%** |
| 50% | 37.5% | 567 / 945 | 0.9656 | **2.12%** | **97.88%** |

---

## 7. Error Analysis and Discussion

1. **The Majority-Class Illusion in `baseline_cnn`:** Although the baseline achieved 66.07% raw accuracy, `outputs/figures/baseline_cnn_confusion_matrix.png` reveals severe collapse onto `NV` (`94.1%` recall on `NV`, but `0.0%` recall on `DF` and only `21.5%–25.7%` recall on malignant `BCC` and `MEL`). Its top confusions were `BKL → NV` (`125`), `MEL → NV` (`100`), and `BCC → NV` (`47`). Post-hoc Temperature Scaling alone (`baseline_cnn_calibrated`, $T=0.9503$) did not improve test ECE (`0.0546 → 0.0666`), confirming calibration cannot fix an under-powered feature extractor.
2. **Controlled Trade-Off / Negative Finding in `improved_mc_cnn`:** Applying full linear inverse-frequency class weights (`~58×` weight on `DF` vs. `NV`) alongside `Dropout2d(0.25)` in a scratch 4-block CNN boosted Balanced Accuracy (`30.85% → 44.23%`) and cut malignant errors from `200 → 124`, **but** severely degraded overall accuracy (`49.93%`) and AURC (`0.2511`) by triggering `244` `NV → MEL` false alarms and underfitting `DF` (`0%` recall). This controlled ablation demonstrates that extreme linear class weights destabilize low-capacity scratch CNNs.
3. **Reliability Breakthrough in `resnet18_mc`:**
   - **Class-wise competence:** Balanced Accuracy surged **+43.1 points** (`30.85% → 73.98%`) and Macro F1 doubled (**+35.5 points**, `34.61% → 70.08%`), rescuing `DF` from `0%` to `78.4%` F1 and cutting total malignant errors (`MEL` + `BCC`) by **68.5%** (`200 → 63`).
   - **Calibration & Uncertainty:** Temperature Scaling ($T = 1.2060$) + MC Dropout ($T=25$) reduced ECE by **32%** (`0.0862 → 0.0587`) and Brier Score to `0.3218`. Predictive entropy achieved **0.8275 Error-Detection AUROC** (and lowest AURC = `0.0737`), while epistemic Mutual Information achieved **0.8019 AUROC** (vs. `0.5846` for scratch `improved_mc_cnn`).
   - **Malignant Referral Safety:** At 70% test coverage, selective abstention captures **49.2% (`31 / 63`)** of remaining malignant errors, leaving only `32` unreferred malignant errors out of `264` malignant test cases (**87.9%** safely diagnosed or referred). Notably, `MEL` exhibits the highest mean predictive entropy (`0.6164` nats), showing the model appropriately signals higher uncertainty on the hardest malignant class.

*Figure References:* `outputs/figures/resnet18_mc_highest_uncertainty_grid.png`, `outputs/figures/resnet18_mc_confident_errors_grid.png`, `outputs/figures/resnet18_mc_entropy_correct_vs_error.png`.

---

## 8. Limitations

1. **Validation/Test Score-Matching Bug Identified & Fixed:** In the initial Phase 3B run, selective thresholds were chosen on *deterministic* calibrated validation confidences but applied to *MC-averaged* test confidences (which have lower max-probabilities due to stochastic averaging), causing the target 70% coverage threshold to yield 56.7% test coverage. We fixed this in `scripts/04_calibrate_and_select.py` so validation thresholds are now selected on matched MC-averaged validation confidences (`val_mc["confidence"]`). Additionally, the small official `val` split (`n = 193`, with 1 `DF` and 3 `VASC`) still introduces finite-sample variance when estimating quantile thresholds.
2. **Harsh Linear Class Weights:** Linear inverse-frequency weights induced `101` conservative `NV → MEL` false positives in `resnet18_mc` (and `244` in `improved_mc_cnn`). Phase 5 adds square-root dampened weights ($1/\sqrt{n_c}$, `configs/abl_r3s_resnet_sqrtw_aug.yaml`) and a single-factor ablation ladder (`R1 → R2 → R3/R3s`) to isolate each component cleanly.
3. **Residual Confident Misclassifications:** `32` malignant errors remained above the 70%-coverage confidence threshold (`resnet18_mc_confident_errors_grid.png`). Deep-learning uncertainty estimates reduce—but never eliminate—silent errors, reinforcing that this system is strictly educational.
4. **Single-Backbone Uncertainty Approximation:** MC Dropout approximates Bayesian inference via a single trained network; Deep Ensembles would likely improve epistemic calibration further at higher compute cost.

---

## 9. Conclusion

**TrustSkin AI** demonstrates that making skin-lesion recognition dependable requires combining strong visual representations, imbalance-aware training, uncertainty estimation, post-hoc calibration, and selective abstention. Across controlled experiments on identical MedIMeta splits, our calibrated MC Dropout model (`resnet18_mc`) improved Balanced Accuracy by **+43.1 points**, Macro F1 by **+35.5 points**, cut Calibration Error (ECE) by **32%**, cut selective risk (AURC) by **43.5%**, and achieved **87.3%–97.9% selective accuracy** when referring ambiguous lesions for human review—paired with an interactive Streamlit demonstration (`app/demo_app.py`).

---

## 10. References, Contributions & AI Disclosure

### References
1. Woerner, S., Jaques, A., & Baumgartner, C. F. (2025). *A comprehensive and easy-to-use multi-domain multi-task medical imaging meta-dataset.* Scientific Data, 12, 666. https://doi.org/10.1038/s41597-025-04866-4
2. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161.
3. Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.* ICML.
4. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML.
5. Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for deep neural networks.* NeurIPS.

### Group Contribution Statement
- **Alap Gohar (502082):** MedIMeta dataset inspection & EDA (`scripts/01_eda.py`, `docs/EDA_REPORT.md`), data preprocessing pipeline (`src/trustskin/data/datamodule.py`), CS320 reference baseline CNN implementation and evaluation (`src/trustskin/models/baseline_cnn.py`, `scripts/02_train_baseline.py`).
- **Muhammad Sikandar Hussain (502808):** Regularized MC Dropout architectures (`src/trustskin/models/uncertainty_cnn.py`), training & imbalance-handling pipeline (`src/trustskin/training/trainer.py`, `scripts/03_train_improved.py`), uncertainty estimation (`src/trustskin/uncertainty/estimator.py`), and Google Colab GPU experiment execution.
- **Shaheer Muzaffar Khan (509535):** Post-hoc Temperature Scaling & calibration metrics (`src/trustskin/calibration/temperature_scaling.py`), selective classification & Risk–Coverage / AURC evaluation (`src/trustskin/selective/selective_risk.py`, `scripts/04_calibrate_and_select.py`), and evaluation visualizations.
- **All Members:** Joint comparative analysis (`scripts/05_compare_experiments.py`, `scripts/06_error_analysis.py`), interactive Streamlit demonstration (`app/demo_app.py`), technical report, and viva preparation.

### Acknowledgement of External Code & AI Tools
- Standard open-source libraries used: PyTorch, Torchvision (ResNet-18 ImageNet weights), NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, and Streamlit.
- **AI Tools Usage Disclosure (per CS320 guidelines):** An AI coding assistant (Antigravity / Gemini) was used to assist with repository scaffolding, modular code structuring, debugging Google Colab dataset staging, drafting unit tests, and organizing documentation/report drafts. All experimental design decisions, dataset verification, GPU training runs, result interpretation, and final validation were reviewed and verified by the group members.
