# TrustSkin AI — Week 17 Presentation & Viva Preparation Guide

**Project:** CS320 P08 — Reliable Recognition of Difficult Skin Lesions  
**Team:** Alap Gohar (502082), Muhammad Sikandar Hussain (502808), Shaheer Muzaffar Khan (509535)

---

## 1. Suggested 10-Slide Presentation Outline (8–10 Minutes)

1. **Title & Motivation:** Why raw accuracy fails in dermoscopy (visually ambiguous lesions + 58.3:1 imbalance); introducing **TrustSkin AI** (classification + uncertainty + calibration + selective referral).
2. **Dataset & EDA Insights (Alap):** MedIMeta `derm` (`11,720` images, 7 classes, `0` split overlap, `58.3:1` imbalance, tiny `val = 193` with 1 `DF` sample). Show `eda_class_distribution.png` & `eda_sample_grid.png`.
3. **CS320 Reference Baseline (Alap):** 3-block scratch CNN (`66.07%` Acc, `30.85%` Bal Acc, `34.61%` Macro F1, `0%` DF recall, `200` malignant errors). Show `baseline_cnn_confusion_matrix.png`.
4. **Proposed Reliability Pipeline (Sikandar & Shaheer):** Regularized MC Dropout ($T=25$) → Post-hoc Temperature Scaling on `val` → Selective Classification (Accept vs. Refer).
5. **Controlled Ablation Results:** Master comparison table across `baseline_cnn`, `baseline_cnn_calibrated`, `improved_mc_cnn`, and `resnet18_mc`.
6. **Honest Trade-Off Analysis (Sikandar):** Why scratch `improved_mc_cnn` gained +13.4% Balanced Accuracy but dropped to 49.9% Accuracy (harsh 58× linear class weights → 244 `NV → MEL` false alarms).
7. **Confidence Calibration Results (Shaheer):** Before vs. After Temperature Scaling ($T = 1.206$) + MC Dropout (`ECE: 0.0862 → 0.0587`, -32%; Brier `0.3218`). Show `resnet18_mc_reliability_before.png` & `_after.png`.
8. **Selective Classification & Malignant Safety (Shaheer):** Risk–Coverage curve (`all_models_risk_coverage.png`), AURC cut by 43.5% (`0.1334 → 0.0754`), 87.3%–97.9% selective accuracy, 49.2% malignant error capture at 70% coverage.
9. **Difficult-Case Visual Error Analysis & Live Demo:** Show `resnet18_mc_highest_uncertainty_grid.png` + `confident_errors_grid.png`, then run a 60-second live demo in `streamlit run app/demo_app.py`.
10. **Limitations & Conclusion:** Tiny `val` threshold shift, linear class-weight trade-off, residual confident errors, educational disclaimer.

---

## 2. High-Probability Viva Questions & Crisp Answers

### Q1: Why did you select checkpoints using `val_loss` instead of `val_macro_f1`?
- **Answer:** Our EDA revealed that the official MedIMeta `val` split has only **193 images**, including just **1 Dermatofibroma (`DF`)** and **3 Vascular (`VASC`)** images. On `val_macro_f1`, flipping that single `DF` image causes a massive ±14-point jump. Continuous `val_loss` aggregates smooth log-likelihood across all 193 samples, giving far more stable model selection while still respecting the official MedIMeta splits.

### Q2: Why did Temperature Scaling help `resnet18_mc` (-32% ECE) but slightly hurt `baseline_cnn`?
- **Answer:** Higher-capacity networks (`resnet18_mc`) tend to be **overconfident** (learned $T = 1.206 > 1$ softens logits, dropping ECE `0.0862 → 0.0621`, and MC Dropout brings it to `0.0587`). In contrast, the small 3-block scratch `baseline_cnn` was underfitting and slightly *under-confident* on `val` ($T = 0.950 < 1$), so sharpening logits slightly increased test ECE (`0.0546 → 0.0666`). Calibration cannot fix a weak representation.

### Q3: Why did `improved_mc_cnn` have lower overall accuracy (`49.9%`) than `baseline_cnn` (`66.1%`)?
- **Answer:** That was an informative controlled trade-off: linear inverse-frequency class weights gave `DF` ~58× the loss weight of `NV`. On a small scratch CNN with `Dropout2d(0.25)`, this heavily penalized `NV` and caused `244` `NV → MEL` false alarms (`NV` recall dropped `94% → 54%`), even though Balanced Accuracy rose (`30.9% → 44.2%`) and Melanoma recall jumped (`25.7% → 66.1%`). Using dampened weights ($1/\sqrt{n_c}$) or pretrained features (`resnet18_mc`) resolves this trade-off.

### Q4: What is the difference between Predictive Entropy and Mutual Information (BALD) in MC Dropout?
- **Answer:** **Predictive Entropy** $H(\bar{p})$ measures **total uncertainty** (both data/aleatoric noise like blurry/ambiguous lesions AND model/epistemic uncertainty). **Mutual Information** $H(\bar{p}) - \mathbb{E}[H(p^{(t)})]$ subtracts average single-pass entropy, isolating **epistemic uncertainty** (how much the stochastic dropout sub-networks disagree with each other). On `resnet18_mc`, both strongly detect errors (`0.8275` and `0.8019` AUROC).

### Q5: How do you ensure zero data leakage during calibration and selective thresholding?
- **Answer:** Temperature Scaling (`TemperatureScaler.fit`) and selective coverage thresholds (`select_threshold_for_coverage`) are fit **exclusively on the `val` split** (`193` images). The `test` split (`1,512` images) is evaluated only once at the end.

### Q6: Why do you freeze BatchNorm during MC Dropout inference?
- **Answer:** Calling `model.train()` globally would make BatchNorm layers use noisy mini-batch statistics instead of learned running mean/variance (and fail for batch size 1!). Our `enable_mc_dropout()` sets the model to `eval()` first, then selectively enables `train()` **only** on `Dropout` / `Dropout2d` layers.

### Q7: How did you isolate the contribution of each component instead of changing 5 things at once?
- **Answer:** We ran a strict single-factor ablation ladder on identical MedIMeta splits: `B0` (scratch baseline, 30.9% Bal Acc) → `R1` (+ pretrained ResNet-18, 53.8%) → `R2` (+ augmentation, 67.3%, AURC 0.0578) → `R3s` (+ sqrt class weights, 70.4% Bal Acc, MEL recall jumps 40.4% → 70.8%, ECE 0.0486) → `Deep Ensemble` (3 seeds of R3s: **75.7% Bal Acc, 74.6% Macro F1, ECE 0.0204**).

### Q8: How did MC Dropout compare against a Deep Ensemble for uncertainty estimation?
- **Answer:** On identical R3s models with Temperature Scaling, MC Dropout ($T=25$) improved ECE slightly over deterministic softmax (`0.0505 → 0.0450`) and gave strong epistemic MI (`0.8043` AUROC). However, a **3-seed Deep Ensemble** outperformed both across every single metric: ECE dropped by more than half to **`0.0204`**, Brier dropped `0.2970 → 0.2815`, AURC dropped `0.0620 → 0.0573`, and Macro F1 jumped `71.04% → 74.58%`.

### Q9: How do you prove statistically that TrustSkin AI genuinely improves difficult cases?
- **Answer:** We defined three difficult subsets up front (`hard_6_non_nv` $n=603$, `confusable_triad_mel_nv_bkl` $n=1{,}297$, and `baseline_failures` $n=513$) and ran 1,000-resample bootstrap 95% CIs plus exact paired **McNemar tests** against the baseline. On `hard_6_non_nv`, our Deep Ensemble lifted accuracy from **23.88% → 73.30%** (+298 net fixes, $p = 7.44 \times 10^{-71}$) and fixed **66.5% (`341/513`)** of baseline errors.

### Q10: Why did you fix the selective-threshold code in Phase 5?
- **Answer:** We discovered a subtle methodological bug in our first run: picking thresholds on *deterministic* validation confidences and applying them to *MC-averaged* test confidences caused a coverage shift (70% target → 56.7% actual) because MC averaging lowers max probabilities. Matching validation and test score types in `04_calibrate_and_select.py` eliminated that mismatch.
