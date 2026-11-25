# TrustSkin AI — Project Progress & Decision Log

**Course:** CS320 — Artificial Neural Networks & Deep Learning (BSAI-2k24, Fall 2026, NUST SEECS)  
**Project ID:** P08 — Reliable Recognition of Difficult Skin Lesions  
**System Name:** TrustSkin AI  
**Team Members:**
1. Alap Gohar (502082) — Dataset Preparation & Baseline Development
2. Muhammad Sikandar Hussain (502808) — Uncertainty Estimation & Model Improvement
3. Shaheer Muzaffar Khan (509535) — Confidence Calibration & Selective Classification

> This log tracks every phase, architectural/experimental decision, dataset discovery, and verification result throughout the project to ensure full reproducibility and directly support the Week 15 Technical Report and Week 17 Viva.

---

## Phase 1 — Repository Architecture & Core Package Scaffolding (Completed: 2026-11-24)

### Key Decisions & Rationale
- **Framework:** Standardized on **PyTorch** (`torch`, `torchvision`) to match the official `medimeta` Python loader and enable straightforward Google Colab GPU training via `pip install -e .`.
- **Modular `src/` Layout:** Separated data loading, models, training, uncertainty estimation, calibration, selective classification, and evaluation into `src/trustskin/` so each group member owns clean, testable modules.
- **Reproducibility:** Centralized all hyperparameters and random seeds (`seed: 42`) in hierarchical YAML configs (`configs/base.yaml`, `configs/baseline_cnn.yaml`, `configs/improved_mc_cnn.yaml`).
- **Strict Split Governance:** Enforced CS320's test-set independence rule in code — `val` is used for model selection, Temperature Scaling fitting, and selective threshold calibration; `test` is reserved strictly for final evaluation.
- **Demo Prototype:** Scaffolded a Streamlit interactive UI (`app/demo_app.py`) for the Week 15 working demonstration.
- **Documentation Organization:** Moved the course specification (`CS320_Fall2026_Term_Project_Final.pdf`) and group proposal (`ANN and DL Project Proposal final.pdf`) into `docs/`.

### Verification
- `python -m compileall -q src scripts app tests` — passed with 0 errors.
- `pytest tests/ -v` — **10/10** synthetic-data unit tests passed (`BaselineCNN`, `UncertaintyCNN` MC Dropout stochasticity, predictive entropy, mutual information, Temperature Scaling NLL reduction, ECE/Brier, Risk–Coverage, AURC).

---

## Phase 2 — Real Dataset Inspection, Alignment & Local EDA (Completed: 2026-11-24)

### Dataset Verification (`data/derm/`)
- Confirmed official MedIMeta Dermatoscopy (`derm`, HAM10000 / ISIC 2018) structure: `11,720` RGB `.tiff` images (`224 × 224 × 3`, `uint8`), `images.hdf5` (1.76 GB), `annotations.csv`, `info.yaml`, `splits/{train,val,test}.txt`, and `task_labels/disease category.npy`.
- Verified **0 null values**, **11,720 unique source images**, and **0 sample overlap** across `train` (10,015), `val` (193), and `test` (1,512).
- Full breakdown recorded in [docs/EDA_REPORT.md](EDA_REPORT.md).

### Code & Config Decisions Driven by Dataset Findings
1. **Exact Task Name & Path Alignment:** Updated `configs/base.yaml` to `data_root: "data"` and `task_name: "disease category"` (exact lowercase key from `data/derm/info.yaml`).
2. **Git Hygiene for Large Data:** Updated `.gitignore` to ignore `data/*` (except `.gitkeep`) so the ~1.8 GB `data/derm/` folder is never committed to Git.
3. **Dual-Mode Dataset Loader:** Extended `src/trustskin/data/datamodule.py` with `LocalMedIMetaDataset` (direct PIL + `.npy` + `info.yaml` fallback) and `get_split_labels()` (fast metadata-only label loading) so the codebase works seamlessly both on Google Colab (with `medimeta`) and locally without `h5py`/`medimeta`.
4. **Checkpoint Selection Metric (`val_loss` instead of `val_macro_f1`):** Because the official MedIMeta `val` split has only **193 samples** with **1 Dermatofibroma (DF)** and **3 Vascular lesion (VASC)** images, `val_macro_f1` exhibits high step-noise (±14% from a single DF sample). Switched checkpoint selection to smoother `val_loss` while continuing to log `val_macro_f1` and `val_balanced_accuracy` every epoch.
5. **Controlled Ablation Configs Added:**
   - `configs/baseline_cnn.yaml` — Exact CS320 3-block scratch CNN (`32 → 64 → 128 → GAP → Linear(7)`), Cross-Entropy, no augmentation.
   - `configs/ablation_weighted_aug.yaml` — `UncertaintyCNN` + dermoscopy augmentations + class-weighted CE (deterministic ablation).
   - `configs/improved_mc_cnn.yaml` — `UncertaintyCNN` + augmentations + class-weighted CE + MC Dropout ($T=25$) + Temperature Scaling.
   - `configs/resnet18_mc.yaml` — Pretrained `ResNet18MC` comparative reference (explicitly comparative, never replacing the CS320 baseline).

### Verification
- `scripts/01_eda.py --config configs/base.yaml` executed locally on `data/derm/`, generating `outputs/tables/eda_class_distribution.csv`, `outputs/tables/eda_summary.json`, `outputs/figures/eda_class_distribution.png`, and `outputs/figures/eda_sample_grid.png`.
- `pytest tests/ -v` — **11/11 passed** (including real-dataset smoke test `tests/test_datamodule_local.py`).
- `git status --short` — confirmed `data/derm/` is completely ignored.

---

## Phase 3A — Pre-Colab Instrumentation, Error Analysis & Turnkey Colab Notebook (Completed: 2026-11-24)

### Upgrades Implemented
- **Error Detection AUROC (`src/trustskin/uncertainty/estimator.py`):** Added `compute_error_detection_auroc()` to quantify how well `1 - confidence`, `predictive_entropy`, and `mutual_information` separate misclassified lesions from correct predictions.
- **Per-Sample Test Prediction Exports (`scripts/02_train_baseline.py`, `scripts/04_calibrate_and_select.py`):** Every evaluated model now saves `outputs/tables/<exp>_test_predictions.csv` and supports `--mc-samples 1` so `BaselineCNN + Temperature Scaling` (`Exp 0-Cal`) can be evaluated as a controlled ablation.
- **Combined Comparison & Convergence Plots (`scripts/05_compare_experiments.py`):** Added multi-model Risk–Coverage overlay (`all_models_risk_coverage.png`) and train/val loss & Macro-F1 curves (`training_curves.png`).
- **Difficult-Case & Malignant-Safety Analysis (`scripts/06_error_analysis.py`):** Computes top confused class pairs, per-class error/entropy summaries, malignant (`MEL`, `BCC`) error capture rate under selective referral (90%/80%/70% coverage), and visual grids of the 12 highest-uncertainty test lesions and 12 most confident misclassifications.
- **Turnkey Google Colab Notebook (`notebooks/00_colab_setup_and_train.ipynb`):** Pre-filled with GitHub URL, automatic direct Zenodo download of MedIMeta `derm` onto Colab SSD (record `7884735`), all 4 controlled experiments (`Exp 0`, `Exp 0-Cal`, `Exp 1/2/3`, `Exp 4`), and automatic Drive backup + zip packaging of `checkpoints/` and `outputs/`.

---

## Phase 3B — Google Colab T4 GPU Controlled Experiments & Local Sync (Completed: 2026-11-25)

### Execution Summary
- Executed `notebooks/00_colab_setup_and_train.ipynb` end-to-end on a **Google Colab Tesla T4 GPU** (`PyTorch 2.14.1+cu130`, CUDA 13.0) using direct 16-stream `aria2c` download of MedIMeta `derm.zip` (Zenodo `7884735`).
- Verified all **12/12** unit + real-dataset tests passed on Colab prior to training.
- Trained and evaluated all 4 controlled experiments on identical MedIMeta splits (`train: 10,015`, `val: 193`, `test: 1,512`), then synced `checkpoints/` (3 verified `.pt` models) and `outputs/` (27 figures, 14 tables/JSONs, 3 logs) back locally.
- Detailed analysis and report-ready tables recorded in [docs/EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

---

## Experiment Registry (Verified Test Set Results, $n = 1{,}512$)

| Exp ID | Config | Description | Test Acc | Bal Acc | Macro F1 | ECE (Before → After) | Brier (After) | AURC ↓ | Err-AUROC (Ent) ↑ | Temp ($T$) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp 0** | `baseline_cnn.yaml` | CS320 3-block CNN (scratch, CE, no aug, best ep 18) | 66.07% | 30.85% | 34.61% | 0.0546 → 0.0546 | 0.4462 | 0.1334 | 0.8110 | 1.0000 | Verified |
| **Exp 0-Cal** | `baseline_cnn.yaml` (`--mc-samples 1`) | Baseline CNN + Post-hoc Temp Scaling on Val | 66.07% | 30.85% | 34.61% | 0.0546 → 0.0666 | 0.4485 | 0.1335 | 0.8123 | 0.9503 | Verified |
| **Exp 1/2/3** | `improved_mc_cnn.yaml` | `UncertaintyCNN` + Aug + Weighted CE + MC ($T=25$) + TS (best ep 19) | 49.93% | 44.23% | 34.28% | 0.0870 → 0.0770 | 0.5724 | 0.2511 | 0.7875 | 0.9646 | Verified |
| **Exp 4** | `resnet18_mc.yaml` | Comparative Pretrained `ResNet18MC` + Weighted CE + MC ($T=25$) + TS (best ep 12) | **77.25%** | **73.98%** | **70.08%** | **0.0862 → 0.0587** | **0.3218** | **0.0754** *(Ent: 0.0737)* | **0.8275** *(MI: 0.8019)* | 1.2060 | Verified (Best) |

---

## Phase 4 — Interactive Demo UI, Week 15 Technical Report & Viva Prep (Completed: 2026-11-25)

### Deliverables Built
- **Pure-PyTorch ResNet-18 Inference Fallback (`src/trustskin/models/uncertainty_cnn.py`):** Added `_SimpleResNet18` matching `torchvision.models.resnet18` `state_dict` keys 1:1 so `checkpoints/resnet18_mc_best.pt` loads locally on CPU without requiring `torchvision`.
- **Interactive 3-Tab Streamlit Demo (`app/demo_app.py`):**
  - *Tab 1 (Live Lesion Triage):* Real held-out MedIMeta test presets (Clear Nevus, Melanoma, Rare Dermatofibroma, Ambiguous High-Uncertainty, Misclassified Melanoma) + custom upload, side-by-side **Baseline CNN vs. TrustSkin AI (ResNet18-MC)** comparison, live MC Dropout + Temperature Scaling ($T = 1.2060$), and interactive Selective Confidence Threshold slider showing Accept vs. Refer decisions.
  - *Tab 2 (Controlled Experiments Dashboard):* Interactive `experiment_comparison.csv` + visual gallery of all key report figures.
  - *Tab 3 (Methodology & Team):* Pipeline summary, team roles, and educational disclaimer.
- **Week 15 Technical Report (`report/TrustSkinAI_Technical_Report.md`):** Complete 10-section report following the exact CS320 specification, including all verified tables, Contribution Statement, and AI Tools Usage Disclosure.
- **Week 17 Viva Prep (`docs/VIVA_PREP.md`):** 10-slide presentation outline + technical Q&A guide.

---

## Phase 5A — Controlled Ablation Ladder, Selective-Threshold Bug Fix & Statistical Rigor (Completed: 2026-11-26)

### Implemented Upgrades
- **Selective-Threshold Score-Matching Fix (`scripts/04_calibrate_and_select.py`):** Fixed the methodology bug where validation thresholds were computed on deterministic confidences and applied to MC-averaged test confidences. Validation thresholds are now computed on matched MC-averaged validation confidences, and the script logs inference-stage ablations (`I1 → I4`) on identical weights.
- **Single-Factor Ablation Ladder (`configs/abl_r1_*`, `abl_r2_*`, `abl_r3s_*`):** Added `power` dampening to `compute_class_weights()` (`1.0` = linear, `0.5` = square-root, normalized to mean `1.0`), added `--seed` / `--exp-name` overrides to `scripts/03_train_improved.py`, removed redundant `configs/ablation_weighted_aug.yaml`, and added ordered ablation-ladder output (`ablation_ladder.csv`) to `scripts/05_compare_experiments.py`.
- **Formal Difficult-Case Criteria & Statistical Testing (`scripts/07_difficult_case_eval.py`):** Evaluates all models on `hard_6_non_nv` (`n=603`), `confusable_triad_mel_nv_bkl` (`n=1,297`), and `baseline_failures` (`n=513`) with 1,000-resample bootstrap 95% CIs and exact paired McNemar tests.
  - *Verified on existing checkpoints:* `resnet18_mc` improves `hard_6_non_nv` accuracy from **23.88% → 71.31%** (AURC `0.7718 → 0.1307`, McNemar +286 net, $p = 6.95 \times 10^{-63}$) and fixes **64.5% (`331/513`)** of baseline errors.
- **Second Uncertainty Method (`scripts/08_ensemble_eval.py`):** Evaluates a 3-seed **Deep Ensemble** against single-model **MC Dropout** and **Deterministic Softmax** on identical splits (`uncertainty_method_comparison.csv`).
- **Phase 5 Colab Notebook (`notebooks/01_phase5_ablations_and_ensemble.ipynb`):** Dedicated 11-cell Colab GPU notebook for R1, R2, R3s, seeds 43 & 44, ensemble comparison, and statistical tables.
- **Verification:** **17 / 17** unit tests passing (`tests/test_phase5_stats.py` added).

---

## Upcoming Phases Checklist
- [x] **Phase 5A (Local Code & Statistical Rigor):** Fixed selective-threshold bug, added difficult-case McNemar + bootstrap evaluation, added R1/R2/R3s configs, ensemble script, and Phase 5 Colab notebook (17/17 tests passing).
- [ ] **Phase 5B (Google Colab GPU):** Push Phase 5A updates to GitHub and run `notebooks/01_phase5_ablations_and_ensemble.ipynb` on Colab T4 (~1.5–2.5 hours) for R1, R2, R3s, seeds 43 & 44, and Deep Ensemble evaluation.
- [ ] **Phase 5C (Final Polish & Submission):** Sync Phase 5B results locally, update `report/TrustSkinAI_Technical_Report.md` and `app/demo_app.py` with the final ablation ladder & ensemble results, and export final PDF report.
