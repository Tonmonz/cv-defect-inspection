"""Heatmap overlays and the report figures in reports/<name>/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE, RED, GREEN, GREY = "#4C78A8", "#E45756", "#54A24B", "#9D9D9D"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def heat_layer(heat: np.ndarray, threshold: float) -> Image.Image:
    """The heatmap alone, as a see-through picture: clear below 0.8x the threshold, solid red from 1.3x."""
    level = np.clip((heat / threshold - 0.8) / 0.5, 0, 1)
    rgba = matplotlib.colormaps["jet"](0.35 + 0.65 * level)
    rgba[..., 3] = level ** 0.7 * 0.75
    return Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")


def overlay(image: Image.Image, heat: np.ndarray, threshold: float) -> Image.Image:
    """Paint the heatmap over the image."""
    return Image.alpha_composite(image.convert("RGBA"), heat_layer(heat, threshold)).convert("RGB")


def _save(fig, path):
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def examples(items: list[dict], threshold: float, title: str, path):
    """items: {"image", "heat", "mask" (or None), "kind", "score"}. Top row input, bottom row heatmap."""
    fig, axes = plt.subplots(2, len(items), figsize=(2.6 * len(items), 5.9), squeeze=False, layout="constrained")
    for col, item in enumerate(items):
        top, bottom = axes[0, col], axes[1, col]
        top.imshow(item["image"])
        if item["mask"] is not None and item["mask"].any():
            top.contour(item["mask"], levels=[0.5], colors="lime", linewidths=1.2)
        flagged = item["score"] > threshold
        correct = flagged == (item["kind"] != "good")
        top.set_title(item["kind"].replace("_", " "), fontsize=10)
        bottom.imshow(overlay(item["image"], item["heat"], threshold))
        bottom.set_title(f"{'DEFECT' if flagged else 'OK'}  score {item['score'] / threshold:.2f}"
                         f"{'' if correct else '  (wrong)'}",
                         color=RED if flagged else GREEN, fontsize=10, fontweight="bold")
        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("input (green = true defect)")
    axes[1, 0].set_ylabel("model heatmap")
    fig.suptitle(title)
    _save(fig, path)


def score_distribution(scores, is_defect, calibration, threshold, path):
    rel, cal = scores / threshold, calibration / threshold
    bins = np.linspace(min(rel.min(), cal.min()) * 0.97, max(rel.max(), 1.1) * 1.02, 40)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(rel[~is_defect], bins=bins, color=GREEN, alpha=0.7, label="good (test)")
    ax.hist(rel[is_defect], bins=bins, color=RED, alpha=0.7, label="defect (test)")
    ax.plot(cal, np.full_like(cal, -0.4), "|", color=BLUE, markersize=12, label="good (calibration)")
    ax.axvline(1.0, color="black", linestyle="--", label="threshold")
    ax.set_xlabel("image score / threshold  (above 1 = flagged as defect)")
    ax.set_ylabel("images")
    ax.set_title("Good and defective parts get different scores")
    ax.legend(fontsize=8)
    _save(fig, path)


def roc(scores, is_defect, auroc: float, path):
    fpr, tpr, _ = roc_curve(is_defect, scores)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"model (AUROC {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", label="coin flip (0.5)")
    ax.set_xlabel("good parts wrongly flagged")
    ax.set_ylabel("defects caught")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def per_kind(rows: list[dict], path):
    """Share flagged for every defect type, and share passed for good parts."""
    labels = [r["kind"].replace("_", " ") for r in rows]
    rates = [r["flagged"] / r["count"] if r["kind"] != "good" else 1 - r["flagged"] / r["count"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(rows) + 1.2))
    ax.barh(labels, rates, color=[GREEN if r >= 0.8 else RED for r in rates])
    for i, r in enumerate(rows):
        ok = r["count"] - r["flagged"] if r["kind"] == "good" else r["flagged"]
        ax.text(rates[i], i, f" {ok}/{r['count']}", va="center")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("defects caught / good parts passed, at the threshold")
    ax.set_title("Result per type (test set)")
    ax.invert_yaxis()
    _save(fig, path)


def summary(rows: list[dict], path):
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 2.5, 3.8))
    image = [r["image_auroc"] for r in rows]
    pixel = [r["pixel_auroc"] if r["pixel_auroc"] is not None else 0 for r in rows]
    ax.bar(x - 0.18, image, 0.36, color=BLUE, label="image AUROC: is this part defective?")
    ax.bar(x + 0.18, pixel, 0.36, color=GREEN, label="pixel AUROC: where is the defect?")
    for i in range(len(rows)):
        ax.text(x[i] - 0.18, image[i], f"{image[i]:.3f}", ha="center", va="bottom", fontsize=8)
        if pixel[i]:
            ax.text(x[i] + 0.18, pixel[i], f"{pixel[i]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [n.replace("_", " ") for n in names])
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("AUROC (1.0 = perfect)")
    ax.set_title("Anomaly detection on MVTec AD (trained on good parts only)")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)
