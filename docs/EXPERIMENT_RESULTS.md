# TrustSkin AI — Verified Experimental Results & Statistical Evidence (Phases 3B & 5)

**Course:** CS320 — Artificial Neural Networks & Deep Learning (BSAI-2k24, Fall 2026)  
**Project ID:** P08 — Reliable Recognition of Difficult Skin Lesions  
**Dataset:** MedIMeta Dermatoscopy (`derm`, task: `"disease category"`, 7 classes, $224 \times 224$ RGB)  
**Evaluation Protocol:** All models trained strictly on `train` ($n = 10{,}015$), checkpoint-selected and temperature-calibrated strictly on `val` ($n = 193$), and evaluated once on the held-out `test` split ($n = 1{,}512$).  
**Hardware:** Google Colab Tesla T4 GPU (`PyTorch 2.14.1+cu130`, CUDA 13.0).

---

## 1. Controlled Single-Factor Ablation Ladder (`B0 → R1 → R2 → R3 / R3s → Deep Ensemble`)

Every step adds **one** controlled design choice on identical MedIMeta splits:

| Step | Experiment | Single Change Added | Accuracy [95% CI] | Balanced Acc [95% CI] | Macro F1 [95% CI] | ECE (Before → After) | Brier ↓ | AURC ↓ | Err-AUROC (Ent) ↑ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | `baseline_cnn` | CS320 3-block scratch CNN (CE, no aug) | 66.07% [63.7, 68.4] | 30.85% [27.7, 34.2] | 34.61% [30.5, 38.6] | 0.0546 → 0.0546 | 0.4462 | 0.1334 | 0.8110 |
| **B0-Cal** | `baseline_cnn_calibrated` | + Temp Scaling on Val ($T=0.950$) | 66.07% [63.7, 68.4] | 30.85% [27.7, 34.2] | 34.61% [30.5, 38.6] | 0.0546 → 0.0666 | 0.4485 | 0.1335 | 0.8123 |
| **S1** | `improved_mc_cnn` | 4-block scratch + Aug + Linear Weighted CE + MC + TS | 49.93% [47.6, 52.4] | 44.23% [40.6, 47.7] | 34.28% [31.5, 36.8] | 0.0870 → 0.0770 | 0.5724 | 0.2511 | 0.7875 |
| **R1** | `abl_r1_resnet_ce_noaug` | + Pretrained ResNet-18 (CE, no aug, MC25+TS) | 76.39% [74.3, 78.6] | 53.77% [49.6, 57.9] | 58.04% [53.4, 62.0] | 0.0952 → 0.0779 | 0.3418 | 0.0747 | 0.8359 |
| **R2** | `abl_r2_resnet_ce_aug` | + Dermoscopy Augmentation (CE, MC25+TS) | **79.83%** [77.8, 81.8] | 67.29% [63.4, 71.4] | 67.90% [64.0, 71.4] | 0.0926 → 0.0726 | 0.2955 | 0.0578 | 0.8389 |
| **R3** | `resnet18_mc` | R2 + Linear Class Weights (`power=1.0`, MC25+TS) | 77.25% [75.1, 79.3] | 73.98% [70.0, 77.7] | 70.08% [66.4, 73.4] | 0.0862 → 0.0587 | 0.3218 | 0.0754 | 0.8275 |
| **R3s** | `abl_r3s_resnet_sqrtw_aug` | R2 + **Sqrt Class Weights** (`power=0.5`, MC25+TS) | 78.84% [76.7, 80.9] | 70.37% [66.2, 74.4] | 70.72% [66.6, 74.0] | 0.0693 → 0.0486 | 0.2971 | 0.0626 | 0.8407 |
| **Ens** | **`deep_ensemble`** | **3-Seed Deep Ensemble of R3s (seeds 42, 43, 44, TS)** | **79.70%** [77.8, 81.8] | **75.70%** [71.7, 79.6] | **74.58%** [71.0, 77.6] | **→ 0.0204** | **0.2815** | **0.0573** | **0.8441** *(Conf: 0.8450)* |

### Multi-Seed Robustness of `R3s` (3 Independent Seeds: 42, 43, 44)
- **Seed 42 (`abl_r3s`):** Acc `78.84%`, Bal Acc `70.37%`, Macro F1 `70.72%`, ECE `0.0486`, AURC `0.0626`
- **Seed 43 (`r3s_seed43`):** Acc `77.58%`, Bal Acc `73.85%`, Macro F1 `71.69%`, ECE `0.0589`, AURC `0.0623`
- **Seed 44 (`r3s_seed44`):** Acc `71.49%`, Bal Acc `68.23%`, Macro F1 `66.31%`, ECE `0.0904`, AURC `0.1002`
- **Single-Seed Mean ± Std:** Acc `75.97% ± 3.93%`, Bal Acc `70.82% ± 2.84%`, Macro F1 `69.57% ± 2.87%`
- **3-Seed Deep Ensemble:** Acc **`79.70%`**, Bal Acc **`75.70%`**, Macro F1 **`74.58%`**, ECE **`0.0204`** (eliminating single-seed variance and outperforming every individual seed!).

---

## 2. Uncertainty Method Comparison (Member 2 Core Comparison)

Comparing three uncertainty estimation methods on identical R3s architecture and splits (`outputs/tables/uncertainty_method_comparison.csv`):

| Uncertainty Method | Accuracy | Balanced Acc | Macro F1 | ECE ↓ | Brier ↓ | AURC (Conf / Ent) ↓ | Err-AUROC (Conf / Ent / MI) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Deterministic Softmax (1 model, TS) | 78.84% | 70.46% | 70.58% | 0.0505 | 0.2980 | 0.0629 / 0.0627 | 0.8388 / 0.8393 / 0.5000 |
| MC Dropout (1 model, $T=25$, TS) | 79.17% | 70.98% | 71.04% | 0.0450 | 0.2970 | 0.0626 / 0.0620 | 0.8342 / 0.8369 / **0.8043** |
| **Deep Ensemble (3 seeds, TS)** | **79.70%** | **75.70%** | **74.58%** | **0.0204** | **0.2815** | **0.0573 / 0.0575** | **0.8450 / 0.8441** / 0.7484 |

---

## 3. Formal Difficult-Case Subset Evaluation & Paired McNemar Tests (CS320 P08)

Defined **before** running Phase 5 (`scripts/07_difficult_case_eval.py`):
1. **`hard_6_non_nv` ($n = 603$):** All 6 non-majority classes (`MEL`, `BCC`, `AKIEC`, `BKL`, `DF`, `VASC`).
2. **`confusable_triad_mel_nv_bkl` ($n = 1{,}297$):** Visually similar pigmented lesions (`MEL`, `NV`, `BKL`).
3. **`baseline_failures` ($n = 513$):** Exact test images misclassified by the CS320 Baseline CNN (analysis only).

| Model | `hard_6_non_nv` Acc / AURC | `confusable_triad` Bal Acc | `baseline_failures` Fixed | McNemar vs. Baseline (`hard_6_non_nv`) | McNemar vs. Baseline (`all_test`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_cnn` | 23.88% / 0.7718 | 49.45% | 0 / 513 (0.0%) | — | — |
| `improved_mc_cnn` | 43.78% / 0.4348 | 51.71% | 169 / 513 (32.9%) | +120 net ($p = 8.4 \times 10^{-17}$) | -244 net ($p = 1.5 \times 10^{-24}$) |
| `abl_r1_resnet_ce_noaug` | 49.59% / 0.4278 | 65.51% | 232 / 513 (45.2%) | +155 net ($p = 1.1 \times 10^{-26}$) | +156 net ($p = 1.6 \times 10^{-19}$) |
| `abl_r2_resnet_ce_aug` | 59.04% / 0.2560 | 66.93% | 281 / 513 (54.8%) | +212 net ($p = 2.3 \times 10^{-42}$) | +208 net ($p = 6.6 \times 10^{-30}$) |
| `abl_r3s_resnet_sqrtw_aug` | 66.67% / 0.1621 | 71.91% | 321 / 513 (62.6%) | +258 net ($p = 4.5 \times 10^{-54}$) | +193 net ($p = 3.3 \times 10^{-20}$) |
| `resnet18_mc` (R3 linear) | 71.31% / 0.1307 | 72.70% | 331 / 513 (64.5%) | +286 net ($p = 7.0 \times 10^{-63}$) | +169 net ($p = 2.1 \times 10^{-14}$) |
| **`deep_ensemble` (3 seeds R3s)** | **73.30% / 0.1068** | **75.02%** | **341 / 513 (66.5%)** | **+298 net ($p = 7.4 \times 10^{-71}$)** | **+206 net ($p = 1.3 \times 10^{-21}$)** |

---

## 4. Malignant Referral Safety (`MEL` + `BCC`, Total = 264 Test Cases)

| Model | Total Malignant Errors (100% Cov) | Caught & Referred @ 90% Cov | Caught & Referred @ 80% Cov | Caught & Referred @ 70% Cov | Unreferred Malignant Errors @ 70% Cov |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_cnn` | 200 | 36 (18.0%) | 74 (37.0%) | 107 (53.5%) | 93 (64.8% safe) |
| `improved_mc_cnn` | 124 | 34 (27.4%) | 62 (50.0%) | 77 (62.1%) | 47 (82.2% safe) |
| `resnet18_mc` | 63 | 14 (22.2%) | 23 (36.5%) | 31 (49.2%) | 32 (87.9% safe) |
| **`deep_ensemble`** | **50** *(-75%)* | **17 (34.0%)** | **28 (56.0%)** | **33 (66.0%)** | **17 (93.6% safe!)** |

Per-class error rates on `deep_ensemble`: `BCC` **10.8%** (89.2% recall), `NV` **16.1%**, `DF` **22.7%** (77.3% recall vs. 0% baseline!), `MEL` **23.4%** (76.6% recall), `AKIEC` **30.2%**, `VASC` **31.4%**, `BKL` **35.5%**.
