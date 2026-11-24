"""Interactive Streamlit Demo Prototype for TrustSkin AI (CS320 P08).

Run locally with:
    streamlit run app/demo_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from trustskin.data import DEFAULT_CLASS_NAMES, build_transforms
from trustskin.models import build_model
from trustskin.models.uncertainty_cnn import enable_mc_dropout
from trustskin.uncertainty import mutual_information, predictive_entropy
from trustskin.utils import get_device, load_config

st.set_page_config(
    page_title="TrustSkin AI — Reliable Skin Lesion Classification",
    page_icon="🩺",
    layout="wide",
)

st.title("TrustSkin AI — Reliable Recognition of Difficult Skin Lesions")
st.caption(
    "CS320 Fall 2026 (P08) Educational Research Prototype — "
    "Not intended for clinical diagnosis or medical use."
)

with st.sidebar:
    st.header("Inference Settings")
    config_path = st.text_input("Config Path", value="configs/improved_mc_cnn.yaml")
    ckpt_path = st.text_input("Checkpoint Path", value="checkpoints/improved_mc_cnn_best.pt")
    temperature = st.number_input("Calibrated Temperature (T)", min_value=0.1, max_value=10.0, value=1.0, step=0.05)
    mc_samples = st.slider("MC Dropout Passes (T)", min_value=1, max_value=50, value=20)
    conf_threshold = st.slider("Selective Confidence Threshold", min_value=0.1, max_value=0.99, value=0.75, step=0.01)


@st.cache_resource
def load_trained_model(cfg_path: str, checkpoint_path: str):
    cfg = load_config(cfg_path)
    device = get_device("auto")
    model = build_model(cfg["model"]).to(device)
    ckpt_file = Path(checkpoint_path)
    loaded = False
    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        loaded = True
    return model, cfg, device, loaded


uploaded_file = st.file_uploader("Upload a dermoscopic skin lesion image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(image, caption="Uploaded Dermoscopic Image", use_container_width=True)

    model, cfg, device, loaded = load_trained_model(config_path, ckpt_path)
    if not loaded:
        st.warning(
            f"Checkpoint `{ckpt_path}` not found yet. Running with randomly initialized weights "
            "(UI preview mode). Train the model on Google Colab and place the `.pt` file in `checkpoints/`."
        )

    tf = build_transforms(
        image_size=cfg["dataset"]["image_size"],
        mean=cfg["dataset"]["normalize_mean"],
        std=cfg["dataset"]["normalize_std"],
        augment=False,
    )
    x = tf(image).unsqueeze(0).to(device)

    enable_mc_dropout(model)
    passes = []
    with torch.no_grad():
        for _ in range(mc_samples):
            logits = model(x) / float(temperature)
            passes.append(F.softmax(logits, dim=-1).cpu().numpy())

    mc_probs = np.stack(passes, axis=0)  # (T, 1, C)
    mean_probs = np.mean(mc_probs, axis=0)[0]
    pred_idx = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_idx])
    entropy = float(predictive_entropy(mean_probs))
    epistemic_mi = float(mutual_information(mc_probs)[0])
    accepted = confidence >= conf_threshold

    with col2:
        st.subheader("Prediction & Reliability Summary")
        if accepted:
            st.success(f"**ACCEPT PREDICTION:** {DEFAULT_CLASS_NAMES[pred_idx]}")
        else:
            st.error(
                f"**REFER TO HUMAN REVIEW (Uncertain):** Top guess is {DEFAULT_CLASS_NAMES[pred_idx]}, "
                f"but confidence ({confidence:.2%}) is below threshold ({conf_threshold:.2%})."
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Calibrated Confidence", f"{confidence:.2%}")
        m2.metric("Predictive Entropy", f"{entropy:.3f} nats")
        m3.metric("Epistemic Uncertainty (MI)", f"{epistemic_mi:.4f}")

        prob_df = pd.DataFrame(
            {"Class": DEFAULT_CLASS_NAMES, "Probability": mean_probs}
        ).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Class"))
else:
    st.info("Upload a dermoscopic image above to test TrustSkin AI's prediction, uncertainty, and selective referral.")
