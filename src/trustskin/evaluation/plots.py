"""Visualization helpers for confusion matrices, calibration, uncertainty, and risk-coverage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from trustskin.calibration.temperature_scaling import compute_ece, reliability_bins


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Save a heatmap confusion matrix."""
    arr = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = np.maximum(arr.sum(axis=1, keepdims=True), 1.0)
        plot_arr = arr / row_sums
        fmt = ".2f"
    else:
        plot_arr = arr
        fmt = ".0f"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        plot_arr,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reliability_diagram(
    probs: np.ndarray,
    targets: np.ndarray,
    save_path: str | Path,
    title: str = "Reliability Diagram",
    n_bins: int = 15,
) -> None:
    """Save a reliability diagram comparing predicted confidence against empirical accuracy."""
    bins = reliability_bins(probs, targets, n_bins=n_bins)
    ece = compute_ece(probs, targets, n_bins=n_bins)
    edges = bins["bin_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.bar(
        centers,
        bins["bin_accuracies"],
        width=width * 0.85,
        alpha=0.8,
        edgecolor="black",
        label=f"Model (ECE = {ece:.4f})",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_uncertainty_histograms(
    uncertainties: np.ndarray,
    correct_mask: np.ndarray,
    save_path: str | Path,
    xlabel: str = "Predictive Entropy (nats)",
    title: str = "Uncertainty Distribution: Correct vs. Incorrect",
) -> None:
    """Compare uncertainty distributions for correct vs. misclassified samples."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        uncertainties[correct_mask],
        bins=25,
        alpha=0.6,
        density=True,
        label="Correct Predictions",
    )
    ax.hist(
        uncertainties[~correct_mask],
        bins=25,
        alpha=0.6,
        density=True,
        label="Misclassified Predictions",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_risk_coverage_comparison(
    curves: dict[str, dict[str, np.ndarray]],
    save_path: str | Path,
    title: str = "Selective Classification: Coverage–Risk Comparison",
) -> None:
    """Plot multiple Coverage-Risk curves on the same axes for controlled comparison."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, curve in curves.items():
        aurc = float(np.mean(curve["risks"]))
        ax.plot(
            curve["coverages"],
            curve["risks"],
            linewidth=2.0,
            label=f"{label} (AURC = {aurc:.4f})",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Coverage (Fraction of Accepted Predictions)")
    ax.set_ylabel("Selective Risk (Error Rate on Accepted)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
