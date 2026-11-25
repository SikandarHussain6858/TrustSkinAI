"""TrustSkin AI — Interactive Streamlit Demonstration Prototype (CS320 Fall 2026, P08).

Run locally with:
    streamlit run app/demo_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure `src/` is importable when launched directly via `streamlit run app/demo_app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trustskin.data import DEFAULT_CLASS_NAMES, LocalMedIMetaDataset, build_transforms
from trustskin.models import build_model
from trustskin.models.uncertainty_cnn import enable_mc_dropout
from trustskin.uncertainty import mutual_information, predictive_entropy
from trustskin.utils import get_device, load_config

SHORT_CODES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MALIGNANT_INDICES = {0, 2}

MODEL_PRESETS = {
    "TrustSkin AI — ResNet18-MC (Best, Calibrated)": {
        "config": "configs/resnet18_mc.yaml",
        "checkpoint": "checkpoints/resnet18_mc_best.pt",
        "temperature": 1.2060,
        "mc_default": 25,
        "supports_mc": True,
        "summary": "Acc 77.25% | Bal Acc 73.98% | Macro F1 70.08% | ECE 0.0587 | AURC 0.0754",
    },
    "CS320 Reference — Baseline CNN (Scratch)": {
        "config": "configs/baseline_cnn.yaml",
        "checkpoint": "checkpoints/baseline_cnn_best.pt",
        "temperature": 1.0000,
        "mc_default": 1,
        "supports_mc": False,
        "summary": "Acc 66.07% | Bal Acc 30.85% | Macro F1 34.61% | ECE 0.0546 | AURC 0.1334",
    },
    "Ablation — Improved MC-CNN (Scratch + Weighted CE)": {
        "config": "configs/improved_mc_cnn.yaml",
        "checkpoint": "checkpoints/improved_mc_cnn_best.pt",
        "temperature": 0.9646,
        "mc_default": 25,
        "supports_mc": True,
        "summary": "Acc 49.93% | Bal Acc 44.23% | Macro F1 34.28% | ECE 0.0770 | AURC 0.2511",
    },
}

st.set_page_config(
    page_title="TrustSkin AI — Reliable Skin Lesion Classification",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_model_bundle(preset_name: str):
    """Load model architecture (without needing torchvision locally) and trained weights."""
    spec = MODEL_PRESETS[preset_name]
    cfg = load_config(PROJECT_ROOT / spec["config"])
    model_cfg = dict(cfg["model"])
    if model_cfg.get("type") == "resnet18_mc":
        model_cfg["pretrained"] = False

    device = get_device("auto")
    model = build_model(model_cfg).to(device)
    ckpt_file = PROJECT_ROOT / spec["checkpoint"]
    loaded = False
    best_epoch = None
    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        loaded = True
        best_epoch = ckpt.get("epoch")
    return model, cfg, device, loaded, best_epoch


@st.cache_data
def get_test_case_presets() -> dict[str, dict]:
    """Curate real held-out MedIMeta test images (clear, malignant, rare, and ambiguous)."""
    derm_dir = PROJECT_ROOT / "data" / "derm"
    pred_csv = PROJECT_ROOT / "outputs" / "tables" / "resnet18_mc_test_predictions.csv"
    if not (derm_dir / "info.yaml").exists() or not pred_csv.exists():
        return {}

    ds = LocalMedIMetaDataset(derm_dir, task_name="disease category", split="test")
    df = pd.read_csv(pred_csv)

    presets: dict[str, dict] = {}
    queries = [
        (
            "Clear Benign Nevus (High Confidence)",
            df[(df["true_label"] == 1) & (df["correct"] == 1)].sort_values("conf_cal", ascending=False),
        ),
        (
            "Malignant Melanoma (Correctly Detected)",
            df[(df["true_label"] == 0) & (df["correct"] == 1)].sort_values("conf_cal", ascending=False),
        ),
        (
            "Rare Dermatofibroma (Baseline Failed, ResNet18 Solved)",
            df[(df["true_label"] == 5) & (df["correct"] == 1)].sort_values("conf_cal", ascending=False),
        ),
        (
            "Ambiguous / High-Uncertainty Lesion (Should Refer)",
            df.sort_values("entropy", ascending=False),
        ),
        (
            "Difficult Misclassified Melanoma (Uncertain Error)",
            df[(df["true_label"] == 0) & (df["correct"] == 0)].sort_values("entropy", ascending=False),
        ),
    ]

    for label, sub in queries:
        if len(sub) > 0:
            row = sub.iloc[0]
            idx = int(row["sample_idx"])
            presets[label] = {
                "sample_idx": idx,
                "path": str(derm_dir / ds.rel_paths[idx]),
                "true_label": int(row["true_label"]),
                "true_name": DEFAULT_CLASS_NAMES[int(row["true_label"])],
            }
    return presets


def run_inference(
    image: Image.Image,
    preset_name: str,
    mc_samples: int,
    temperature: float,
) -> dict:
    """Run deterministic or MC Dropout calibrated inference on one RGB PIL image."""
    model, cfg, device, loaded, best_epoch = load_model_bundle(preset_name)
    spec = MODEL_PRESETS[preset_name]
    effective_passes = mc_samples if spec["supports_mc"] else 1

    tf = build_transforms(
        image_size=cfg["dataset"]["image_size"],
        mean=cfg["dataset"]["normalize_mean"],
        std=cfg["dataset"]["normalize_std"],
        augment=False,
    )
    x = tf(image).unsqueeze(0).to(device)

    if effective_passes > 1:
        enable_mc_dropout(model)
    else:
        model.eval()

    passes = []
    with torch.no_grad():
        for _ in range(effective_passes):
            logits = model(x) / float(temperature)
            passes.append(F.softmax(logits, dim=-1).cpu().numpy())

    mc_probs = np.stack(passes, axis=0)  # (T, 1, 7)
    mean_probs = np.mean(mc_probs, axis=0)[0]
    std_probs = np.std(mc_probs, axis=0)[0]
    pred_idx = int(np.argmax(mean_probs))

    return {
        "loaded": loaded,
        "best_epoch": best_epoch,
        "passes": effective_passes,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "pred_idx": pred_idx,
        "pred_name": DEFAULT_CLASS_NAMES[pred_idx],
        "confidence": float(mean_probs[pred_idx]),
        "entropy": float(predictive_entropy(mean_probs)),
        "epistemic_mi": float(mutual_information(mc_probs)[0]) if effective_passes > 1 else 0.0,
    }


def render_prediction_card(
    title: str,
    result: dict,
    conf_threshold: float,
    true_label_name: str | None = None,
) -> None:
    """Render triage decision, reliability metrics, and probability breakdown."""
    st.markdown(f"### {title}")
    if not result["loaded"]:
        st.warning("Checkpoint not found — running with untrained weights (preview mode).")

    accepted = result["confidence"] >= conf_threshold
    pred_idx = result["pred_idx"]
    is_malignant = pred_idx in MALIGNANT_INDICES

    if accepted:
        if is_malignant:
            st.warning(
                f"**ACCEPT PREDICTION (Malignant Class):** `{result['pred_name']}` "
                f"(Calibrated Confidence: **{result['confidence']:.1%}** ≥ {conf_threshold:.0%})"
            )
        else:
            st.success(
                f"**ACCEPT PREDICTION:** `{result['pred_name']}` "
                f"(Calibrated Confidence: **{result['confidence']:.1%}** ≥ {conf_threshold:.0%})"
            )
    else:
        st.error(
            f"**REFER TO DERMATOLOGIST (UNCERTAIN):** Top guess is `{result['pred_name']}`, "
            f"but calibrated confidence (**{result['confidence']:.1%}**) is below threshold (**{conf_threshold:.0%}**)."
        )

    if true_label_name is not None:
        match_icon = "✅ Correct" if result["pred_name"] == true_label_name else "❌ Misclassified"
        st.caption(f"Held-out MedIMeta Ground Truth: **{true_label_name}** ({match_icon})")

    c1, c2, c3 = st.columns(3)
    c1.metric("Calibrated Confidence", f"{result['confidence']:.1%}")
    c2.metric("Predictive Entropy", f"{result['entropy']:.3f} nats")
    c3.metric("Epistemic MI (BALD)", f"{result['epistemic_mi']:.4f}")

    prob_df = pd.DataFrame(
        {
            "Lesion Class": [f"{SHORT_CODES[i]} — {DEFAULT_CLASS_NAMES[i]}" for i in range(7)],
            "Probability": result["mean_probs"],
        }
    ).sort_values("Probability", ascending=False)
    st.bar_chart(prob_df.set_index("Lesion Class"), height=230)


# --- Header ---
st.title("🩺 TrustSkin AI — Reliable Recognition of Difficult Skin Lesions")
st.caption(
    "CS320: Artificial Neural Networks & Deep Learning (BSAI-2k24, Fall 2026) — Project P08 | "
    "**Educational Research Prototype Only — Not Intended for Clinical Diagnosis or Medical Use**"
)

tab_triage, tab_dashboard, tab_about = st.tabs(
    [
        "🔬 Live Lesion Triage",
        "📊 Controlled Experiments & Visual Evidence",
        "📘 Methodology & Team",
    ]
)

# ==============================================================================
# TAB 1: LIVE SKIN LESION TRIAGE
# ==============================================================================
with tab_triage:
    with st.sidebar:
        st.header("Triage Controls")
        selected_model = st.selectbox("Primary Model", list(MODEL_PRESETS.keys()), index=0)
        compare_baseline = st.checkbox(
            "Side-by-Side: Compare vs. Baseline CNN",
            value=True,
            disabled=(selected_model == "CS320 Reference — Baseline CNN (Scratch)"),
        )
        preset = MODEL_PRESETS[selected_model]
        st.caption(preset["summary"])

        mc_samples = st.slider(
            "MC Dropout Passes (T)",
            min_value=5,
            max_value=40,
            value=preset["mc_default"] if preset["supports_mc"] else 25,
            step=5,
            disabled=not preset["supports_mc"],
        )
        temperature = st.number_input(
            "Calibrated Temperature (T)",
            min_value=0.1,
            max_value=5.0,
            value=float(preset["temperature"]),
            step=0.05,
        )
        conf_threshold = st.slider(
            "Selective Confidence Threshold",
            min_value=0.25,
            max_value=0.98,
            value=0.73,
            step=0.01,
            help="Predictions below this calibrated confidence are referred to human review. "
            "On test set: 0.73 ≈ 71% coverage / 87.3% accuracy; 0.84 ≈ 57% coverage / 91.8% accuracy.",
        )

    test_presets = get_test_case_presets()
    input_mode = st.radio(
        "Image Source",
        ["Pick a Held-Out MedIMeta Test Lesion", "Upload Custom Dermoscopic Image"],
        horizontal=True,
    )

    image: Image.Image | None = None
    true_label_name: str | None = None
    image_caption = ""

    if input_mode == "Pick a Held-Out MedIMeta Test Lesion":
        if test_presets:
            choice = st.selectbox("Select Real Test Case", list(test_presets.keys()))
            meta = test_presets[choice]
            image = Image.open(meta["path"]).convert("RGB")
            true_label_name = meta["true_name"]
            image_caption = f"MedIMeta Test Sample #{meta['sample_idx']} — {choice}"
        else:
            st.info("Local `data/derm/` or test predictions CSV not found. Switch to Upload mode.")
    else:
        uploaded = st.file_uploader("Upload a dermoscopic skin lesion image", type=["png", "jpg", "jpeg", "tiff"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            image_caption = f"Uploaded: {uploaded.name}"

    if image is not None:
        img_col, res_col = st.columns([0.9, 2.1])
        with img_col:
            st.image(image, caption=image_caption, use_container_width=True)
            st.info(
                "**Selective Triage Operating Guide (ResNet18-MC Test Set):**\n"
                "- Threshold `0.58` → 85% coverage, **83.5%** selective acc\n"
                "- Threshold `0.73` → 71% coverage, **87.3%** selective acc\n"
                "- Threshold `0.84` → 57% coverage, **91.8%** selective acc\n"
                "- Threshold `0.97` → 38% coverage, **97.9%** selective acc"
            )

        with res_col:
            primary_res = run_inference(image, selected_model, mc_samples, temperature)
            if compare_baseline and selected_model != "CS320 Reference — Baseline CNN (Scratch)":
                baseline_res = run_inference(
                    image,
                    "CS320 Reference — Baseline CNN (Scratch)",
                    mc_samples=1,
                    temperature=1.0,
                )
                left, right = st.columns(2)
                with left:
                    render_prediction_card("TrustSkin AI (Selected)", primary_res, conf_threshold, true_label_name)
                with right:
                    render_prediction_card("CS320 Baseline CNN", baseline_res, conf_threshold, true_label_name)
            else:
                render_prediction_card(selected_model, primary_res, conf_threshold, true_label_name)
    else:
        st.info("Select a held-out test lesion or upload an image above to run TrustSkin AI inference.")

# ==============================================================================
# TAB 2: CONTROLLED EXPERIMENTS & VISUAL EVIDENCE DASHBOARD
# ==============================================================================
with tab_dashboard:
    st.subheader("Controlled Comparison on Strictly Held-Out Test Split (n = 1,512)")
    comp_path = PROJECT_ROOT / "outputs" / "tables" / "experiment_comparison.csv"
    if comp_path.exists():
        st.dataframe(pd.read_csv(comp_path), use_container_width=True, hide_index=True)
    else:
        st.warning("`outputs/tables/experiment_comparison.csv` not found.")

    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    gallery_choice = st.selectbox(
        "Inspect Report Figure",
        [
            "Risk–Coverage Comparison Across All Models",
            "Training & Validation Curves",
            "ResNet18-MC Reliability: Before vs. After Temperature Scaling",
            "Confusion Matrices: Baseline CNN vs. ResNet18-MC",
            "Difficult Cases: Highest-Uncertainty Test Lesions (ResNet18-MC)",
            "Failure Mode Analysis: Most Confident Errors (ResNet18-MC)",
            "Dataset EDA: Class Imbalance & Representative Samples",
        ],
    )

    if gallery_choice == "Risk–Coverage Comparison Across All Models":
        st.image(str(fig_dir / "all_models_risk_coverage.png"), use_container_width=True)
    elif gallery_choice == "Training & Validation Curves":
        st.image(str(fig_dir / "training_curves.png"), use_container_width=True)
    elif gallery_choice == "ResNet18-MC Reliability: Before vs. After Temperature Scaling":
        c1, c2 = st.columns(2)
        c1.image(str(fig_dir / "resnet18_mc_reliability_before.png"), caption="Before Calibration (ECE = 0.0862)", use_container_width=True)
        c2.image(str(fig_dir / "resnet18_mc_reliability_after.png"), caption="After Temp Scaling + MC25 (ECE = 0.0587)", use_container_width=True)
    elif gallery_choice == "Confusion Matrices: Baseline CNN vs. ResNet18-MC":
        c1, c2 = st.columns(2)
        c1.image(str(fig_dir / "baseline_cnn_confusion_matrix.png"), caption="Baseline CNN (Macro F1 = 34.6%)", use_container_width=True)
        c2.image(str(fig_dir / "resnet18_mc_confusion_matrix.png"), caption="ResNet18-MC (Macro F1 = 70.1%)", use_container_width=True)
    elif gallery_choice == "Difficult Cases: Highest-Uncertainty Test Lesions (ResNet18-MC)":
        st.image(str(fig_dir / "resnet18_mc_highest_uncertainty_grid.png"), use_container_width=True)
    elif gallery_choice == "Failure Mode Analysis: Most Confident Errors (ResNet18-MC)":
        st.image(str(fig_dir / "resnet18_mc_confident_errors_grid.png"), use_container_width=True)
    elif gallery_choice == "Dataset EDA: Class Imbalance & Representative Samples":
        st.image(str(fig_dir / "eda_class_distribution.png"), use_container_width=True)
        st.image(str(fig_dir / "eda_sample_grid.png"), use_container_width=True)

# ==============================================================================
# TAB 3: METHODOLOGY, TEAM & DISCLAIMER
# ==============================================================================
with tab_about:
    st.subheader("System Pipeline & Reliability Criteria")
    st.markdown(
        """
        **TrustSkin AI** addresses CS320 Project **P08 (Reliable Recognition of Difficult Skin Lesions)** on the
        standardized **MedIMeta Dermatoscopy** dataset (`11,720` images across 7 diagnostic classes, `58.3 : 1` class imbalance):

        1. **Strict Split Governance:** Trained strictly on `train` (`10,015`), model-selected and calibrated strictly on `val` (`193`), and evaluated once on `test` (`1,512`).
        2. **CS320 Reference Baseline:** 3-block scratch CNN (`32 → 64 → 128 → GAP → Linear(7)`), Cross-Entropy, Adam `1e-3`, 20 epochs.
        3. **Uncertainty Estimation:** Monte Carlo Dropout ($T = 25$ stochastic passes) computing predictive entropy $H(\\bar{p})$ and epistemic mutual information (BALD).
        4. **Post-Hoc Calibration:** Single-parameter Temperature Scaling ($T = 1.2060$ fit via L-BFGS on `val`), reducing ResNet18-MC ECE by **32%** (`0.0862 → 0.0587`).
        5. **Selective Classification:** Abstaining on low-confidence / high-entropy lesions cuts AURC by **43.5%** (`0.1334 → 0.0754`) and lifts accuracy on accepted predictions up to **91.8%–97.9%**.

        ### Group Members (NUST SEECS — BSAI-2k24)
        - **Alap Gohar (502082):** Dataset Preparation, EDA & Baseline CNN Development
        - **Muhammad Sikandar Hussain (502808):** Uncertainty Estimation (MC Dropout) & Model Improvement
        - **Shaheer Muzaffar Khan (509535):** Confidence Calibration (Temperature Scaling) & Selective Classification

        > **Educational Disclaimer:** This system is an academic machine-learning prototype for CS320 and is **not** validated or intended for clinical diagnosis or patient care.
        """
    )
