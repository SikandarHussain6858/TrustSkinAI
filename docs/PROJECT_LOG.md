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

## Experiment Registry (To Be Populated During Phase 3B — Google Colab GPU Runs)

| Exp ID | Config | Description | Test Acc | Bal Acc | Macro F1 | ECE (Before → After) | Brier (After) | AURC | Err-AUROC (Ent) | Temp ($T$) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp 0** | `baseline_cnn.yaml` | CS320 3-block CNN (scratch, CE, no aug) | — | — | — | — | — | — | — | 1.0 | Ready for Colab |
| **Exp 0-Cal** | `baseline_cnn.yaml` (`--mc-samples 1`) | Baseline CNN + Post-hoc Temp Scaling on Val | — | — | — | — | — | — | — | — | Ready for Colab |
| **Exp 1/2/3** | `improved_mc_cnn.yaml` | `UncertaintyCNN` + Aug + Weighted CE + MC ($T=25$) + Temp Scaling | — | — | — | — | — | — | — | — | Ready for Colab |
| **Exp 4** | `resnet18_mc.yaml` | Comparative Pretrained `ResNet18MC` + MC ($T=25$) + Temp Scaling | — | — | — | — | — | — | — | — | Ready for Colab |

---

## Upcoming Phases Checklist
- [ ] **Phase 3 (Google Colab GPU):** Push repo to GitHub, stage `data/derm` on Colab, run Exp 0 (`baseline_cnn`) and record difficult/confused classes.
- [ ] **Phase 4 (Google Colab GPU):** Run Exp 1–4 controlled ablations, MC Dropout uncertainty evaluation, Temperature Scaling calibration, and Selective Classification Risk–Coverage analysis.
- [ ] **Phase 5 (Local Analysis & UI):** Sync checkpoints & outputs back locally, run difficult-case visual error analysis, verify Streamlit UI (`app/demo_app.py`) with trained weights.
- [ ] **Phase 6 (Week 15 Submission):** Finalize Technical Report (10 sections + Contribution Statement + AI Tool Disclosure) and Week 17 Viva slides.
