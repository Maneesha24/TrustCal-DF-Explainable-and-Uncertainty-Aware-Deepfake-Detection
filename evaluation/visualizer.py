"""
evaluation/visualizer.py
-------------------------
Publication-quality visualisations for TrustCal-DF results.

Plots generated:
  1. Reliability diagram (calibration curve)
  2. Risk-coverage curve
  3. Grad-CAM heatmap grid (original + overlays + perturbed)
  4. Trust score distribution (real vs fake)
  5. Training history curves
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

# Project root (parent of ``evaluation/``).
BASE_DIR = Path(__file__).resolve().parent.parent
# Directory where all evaluation figures are written.
FIG_DIR  = BASE_DIR / "visualizations"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Global matplotlib style for publication-quality figures.
plt.rcParams.update({
    "figure.dpi":      150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":     "DejaVu Sans",
})
# Colour keys used across plots (real / fake / accent / neutral).
PALETTE = {"real": "#2ecc71", "fake": "#e74c3c", "blue": "#3498db", "grey": "#95a5a6"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Reliability diagram
# ══════════════════════════════════════════════════════════════════════════════

def plot_reliability_diagram(
    bin_confs:  np.ndarray,
    bin_accs:   np.ndarray,
    bin_counts: np.ndarray,
    ece:        float,
    save:       bool = True,
    tag:        str  = "",
) -> plt.Figure:
    """
    Draw a reliability (calibration) diagram with per-bin sample counts.

    Left panel: confidence vs accuracy bars, perfect-calibration diagonal, and
    gap fill (red = overconfident, green = underconfident). Right panel: horizontal
    histogram of samples per bin.

    @param bin_confs: Mean predicted confidence per ECE bin, shape ``(n_bins,)``.
    @param bin_accs: Mean empirical accuracy per bin, shape ``(n_bins,)``.
    @param bin_counts: Number of samples falling in each bin, shape ``(n_bins,)``.
    @param ece: Expected Calibration Error (shown in title).
    @param save: If ``True``, write PNG to ``visualizations/reliability_diagram{tag}.png``.
    @param tag: Optional suffix for the output filename.
    @return: Matplotlib ``Figure`` handle.
    @see expected_calibration_error (in ``evaluation.trust_score``)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"], lw=1.5, label="Perfect")

    # Gap fill
    n_bins = len(bin_confs)
    width  = 1.0 / n_bins
    xs     = np.array([b * width + width / 2 for b in range(n_bins)])

    # Confidence bars (behind the accuracy bars)
    ax.bar(xs, bin_confs, width=width * 0.9, alpha=0.25,
           color=PALETTE["blue"], label="Confidence")
    # Accuracy bars
    ax.bar(xs, bin_accs, width=width * 0.9, alpha=0.7,
           color=PALETTE["blue"], label="Accuracy")

    # Gap (red = overconfident, green = underconfident)
    for x, conf, acc in zip(xs, bin_confs, bin_accs):
        color  = PALETTE["fake"] if conf > acc else PALETTE["real"]
        bottom = min(conf, acc)
        height = abs(conf - acc)
        ax.bar(x, height, bottom=bottom, width=width * 0.9,
               alpha=0.4, color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (predicted probability of fake)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram  —  ECE = {ece:.4f}")
    ax.legend(loc="upper left", fontsize=8)

    # Sample count histogram
    ax2 = axes[1]
    ax2.barh(xs, bin_counts, height=width * 0.85, color=PALETTE["blue"], alpha=0.7)
    ax2.set_xlabel("Samples")
    ax2.set_title("Bin counts")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if save:
        path = FIG_DIR / f"reliability_diagram{tag}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[Viz] Saved: {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Risk-coverage curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_risk_coverage(
    coverages:  np.ndarray,
    risks:      np.ndarray,
    thresholds: np.ndarray,
    save:       bool = True,
    tag:        str  = "",
) -> plt.Figure:
    """
    Plot selective risk against coverage (trust-threshold sweep).

    Curve is coloured by abstention threshold τ. Lower risk at higher coverage
    indicates a better trust-scoring mechanism.

    @param coverages: Fraction of samples answered at each τ, shape ``(n,)``.
    @param risks: Selective error rate on covered samples, shape ``(n,)``.
    @param thresholds: Trust thresholds used for the sweep, shape ``(n,)``.
    @param save: If ``True``, write PNG to ``visualizations/risk_coverage{tag}.png``.
    @param tag: Optional filename suffix.
    @return: Matplotlib ``Figure`` handle.
    @see risk_coverage_curve (in ``evaluation.trust_score``)
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Colour the curve by threshold value
    from matplotlib.collections import LineCollection
    points = np.array([coverages, risks]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc     = LineCollection(segs, cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    lc.set_array(thresholds[:-1])
    lc.set_linewidth(2.5)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="Trust threshold τ")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(risks.max(), 0.1) * 1.1)
    ax.set_xlabel("Coverage (fraction of samples answered)")
    ax.set_ylabel("Selective Risk (error rate on answered)")
    ax.set_title("Risk-Coverage Curve")
    ax.axhline(risks[0], color=PALETTE["grey"], ls="--", lw=1,
               label=f"Full coverage risk = {risks[0]:.3f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        path = FIG_DIR / f"risk_coverage{tag}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[Viz] Saved: {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Grad-CAM heatmap grid
# ══════════════════════════════════════════════════════════════════════════════

def plot_gradcam_grid(
    image_tensor,              # (1, C, H, W) normalised torch.Tensor
    orig_heatmap: np.ndarray,  # (H, W) float32
    perturbed_heatmaps: List[np.ndarray],
    consistency: float,
    label: int,
    prediction: int,
    save: bool = True,
    tag:  str  = "",
) -> plt.Figure:
    """
    Visualise Grad-CAM stability for one image in a horizontal grid.

    Columns: original RGB | JET overlay on original | perturbed saliency maps.

    @param image_tensor: Normalised input, shape ``(1, 3, H, W)`` (torch.Tensor).
    @param orig_heatmap: Grad-CAM for the unperturbed image, ``(H, W)`` in ``[0, 1]``.
    @param perturbed_heatmaps: List of heatmaps from noisy inputs (length ``K``).
    @param consistency: Mean SSIM explanation consistency (shown in subtitle).
    @param label: Ground-truth class (0 = real, 1 = fake).
    @param prediction: Model prediction (0/1).
    @param save: If ``True``, write PNG to ``visualizations/gradcam_grid{tag}.png``.
    @param tag: Optional filename suffix.
    @return: Matplotlib ``Figure`` handle.
    @see explanation_consistency (in ``explainability.gradcam``)
    """
    import torch
    import cv2

    # Denormalize image
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])
    img  = image_tensor[0].permute(1, 2, 0).numpy()
    img  = (img * STD + MEAN).clip(0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    # Grad-CAM overlay
    hm_uint8 = (orig_heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay  = (0.5 * hm_color + 0.5 * img_uint8).astype(np.uint8)

    K = len(perturbed_heatmaps)
    n_cols = 2 + K
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))

    # Original
    axes[0].imshow(img_uint8)
    axes[0].set_title(
        f"Image\nlabel={'fake' if label else 'real'} "
        f"pred={'fake' if prediction else 'real'}",
        fontsize=8
    )

    # Grad-CAM overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM\nconsistency={consistency:.3f}", fontsize=8)

    # Perturbed heatmaps
    for k, h in enumerate(perturbed_heatmaps):
        axes[2 + k].imshow(h, cmap="jet", vmin=0, vmax=1)
        axes[2 + k].set_title(f"Perturb {k+1}", fontsize=8)

    for ax in axes:
        ax.axis("off")

    plt.suptitle("Grad-CAM Explanation Consistency", fontsize=10, y=1.02)
    plt.tight_layout()

    if save:
        path = FIG_DIR / f"gradcam_grid{tag}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[Viz] Saved: {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Trust score distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_trust_distribution(
    trust_scores: np.ndarray,
    labels:       np.ndarray,
    threshold:    float = 0.5,
    save:         bool  = True,
    tag:          str   = "",
) -> plt.Figure:
    """
    KDE of trust scores separated by ground-truth class.

    Overlays a vertical line at the selective-prediction threshold τ.

    @param trust_scores: Per-sample trust, shape ``(N,)`` in ``[0, 1]``.
    @param labels: Binary labels (0 = real, 1 = fake), shape ``(N,)``.
    @param threshold: Abstention cutoff τ (dashed vertical line).
    @param save: If ``True``, write PNG to ``visualizations/trust_distribution{tag}.png``.
    @param tag: Optional filename suffix.
    @return: Matplotlib ``Figure`` handle.
    @see compute_trust_scores (in ``evaluation.trust_score``)
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for lbl, name, color in [(0, "Real", PALETTE["real"]),
                              (1, "Fake", PALETTE["fake"])]:
        mask = labels == lbl
        if mask.sum() < 2:
            continue
        sns.kdeplot(
            trust_scores[mask], ax=ax,
            fill=True, alpha=0.35, color=color, label=name,
        )

    ax.axvline(threshold, color="black", ls="--", lw=1.5,
               label=f"Threshold τ={threshold}")
    ax.set_xlabel("Trust Score")
    ax.set_ylabel("Density")
    ax.set_title("Trust Score Distribution by Class")
    ax.legend()
    plt.tight_layout()

    if save:
        path = FIG_DIR / f"trust_distribution{tag}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[Viz] Saved: {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Training history
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(
    history: List[Dict],
    save:    bool = True,
    tag:     str  = "",
) -> plt.Figure:
    """
    Plot training curves from the JSON history written by the trainer.

    Three panels: train/val accuracy, validation loss, validation AUROC vs epoch.

    @param history: List of per-epoch dicts (keys: ``epoch``, ``train_acc``,
                    ``val_loss``, ``val_acc``, ``val_auroc``).
    @param save: If ``True``, write PNG to ``visualizations/training_history{tag}.png``.
    @param tag: Optional filename suffix.
    @return: Matplotlib ``Figure`` handle.
    @see train (in ``training.trainer``)
    """
    epochs    = [h["epoch"]     for h in history]
    val_loss  = [h["val_loss"]  for h in history]
    val_acc   = [h["val_acc"]   for h in history]
    val_auroc = [h["val_auroc"] for h in history]
    train_acc = [h["train_acc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(epochs, train_acc, label="Train acc", color=PALETTE["blue"])
    axes[0].plot(epochs, val_acc,   label="Val acc",   color=PALETTE["real"])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy"); axes[0].legend()

    axes[1].plot(epochs, val_loss, color=PALETTE["fake"])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")

    axes[2].plot(epochs, val_auroc, color=PALETTE["blue"])
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUROC")
    axes[2].set_title("Validation AUROC")

    plt.tight_layout()
    if save:
        path = FIG_DIR / f"training_history{tag}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[Viz] Saved: {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: render all plots from a results dict
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(report: Dict, curve: Dict, history: Optional[List] = None, tag: str = ""):
    """
    Generate the standard set of evaluation figures in one call.

    Renders reliability diagram, risk–coverage curve, and trust distribution;
    optionally includes training history if ``history`` is provided.

    @param report: Output of ``full_report`` (must include ``_bin_*``, ``_trust_scores``,
                   ``_labels``, ``ece``, ``threshold``).
    @param curve: Output of ``risk_coverage_curve`` (``coverages``, ``risks``, ``thresholds``).
    @param history: Optional training history list; skipped if ``None``.
    @param tag: Filename suffix passed to each sub-plot function.
    @return: None (figures saved under ``FIG_DIR`` when ``save=True`` in sub-calls).
    @see plot_reliability_diagram
    @see plot_risk_coverage
    @see plot_trust_distribution
    @see plot_training_history
    """
    plot_reliability_diagram(
        report["_bin_confs"], report["_bin_accs"],
        report["_bin_counts"], report["ece"], tag=tag,
    )
    plot_risk_coverage(
        curve["coverages"], curve["risks"], curve["thresholds"], tag=tag,
    )
    plot_trust_distribution(
        report["_trust_scores"], report["_labels"],
        threshold=report["threshold"], tag=tag,
    )
    if history:
        plot_training_history(history, tag=tag)

    print("[Viz] All plots saved to", FIG_DIR)