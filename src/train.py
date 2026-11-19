"""Learn what good parts look like from images, then find defects with a heatmap.

    python -m src.train                                 # MVTec AD bottle, metal_nut, screw (downloads them)
    python -m src.train --categories zipper transistor  # other MVTec AD categories
    python -m src.train --data-dir D:/wd_abs --name wd_abs --no-crop   # your own folder (layout: src/data.py)
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from . import plots
from .data import DEFAULT_CATEGORIES, download_mvtec, load_folder
from .align import alignment
from .patchcore import IMAGE_SIZE, FeatureExtractor, PatchCore, load_mask, load_rgb, view
from .regions import defect_regions, located
from .paths import ARTIFACTS_DIR, REPORTS_DIR, ROOT

SEED = 0
ALIGN = []  # products whose parts are turned to one angle with OpenCV first (see README)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES, help="MVTec AD categories")
    ap.add_argument("--data-dir", help="your own image folder instead of MVTec AD")
    ap.add_argument("--name", help="name for --data-dir results (default: the folder name)")
    ap.add_argument("--no-crop", action="store_true",
                    help="keep the whole image (default crops the border, as in MVTec AD benchmarks)")
    ap.add_argument("--coreset", type=float, default=0.01, help="share of good patches to remember")
    ap.add_argument("--max-train", type=int, default=400, help="at most this many good images to learn from")
    ap.add_argument("--calibration", type=float, default=0.1,
                    help="share of good training images kept aside to set the threshold")
    ap.add_argument("--align", nargs="*", default=ALIGN,
                    help="products to align with OpenCV before the model, e.g. --align screw")
    return ap.parse_args(argv)


def stored(path: Path) -> str:
    """Paths inside the project are stored relative to it, so the folder can move."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def run(name: str, root: Path, source: str, crop: bool, coreset: float, max_train: int,
        calibration_share: float, extractor: FeatureExtractor, align: bool = False) -> dict:
    train, test = load_folder(root)
    shuffled = train[:]
    random.Random(SEED).shuffle(shuffled)
    n_cal = max(5, round(len(shuffled) * calibration_share))
    calibration, learn = shuffled[:n_cal], shuffled[n_cal:][:max_train]
    print(f"\n=== {name}: learn from {len(learn)} good images, calibrate on {len(calibration)}, "
          f"test on {len(test)} ({sum(s.is_defect for s in test)} defective)")

    model = PatchCore(extractor, crop=crop, coreset_ratio=coreset, align=align)
    started = time.time()
    info = model.fit(learn)
    fit_seconds = time.time() - started
    print(f"    memory bank: {info['memory_bank']:,} of {info['patches']:,} good patches ({fit_seconds:.0f} s)")

    # The threshold comes from good images only, like on a real line: the highest score a good
    # calibration image got. Everything scoring above it is called a defect.
    cal_scores, _ = model.predict(calibration)
    threshold = float(cal_scores.max())

    started = time.time()
    scores, maps = model.predict([s.path for s in test])
    ms_per_image = (time.time() - started) / len(test) * 1000
    is_defect = np.array([s.is_defect for s in test])
    flagged = scores > threshold

    masks = [load_mask(s.mask, crop, alignment(load_rgb(s.path)) if align else None) if s.mask else None
             for s in test]
    # OpenCV turns each flagged heatmap into boxes; a defect counts as located when the largest
    # box overlaps the real defect in the ground-truth mask.
    regions = [defect_regions(m, threshold) for m in maps]
    caught = [i for i, s in enumerate(test) if s.is_defect and flagged[i] and masks[i] is not None]
    located_count = sum(located(regions[i], masks[i]) for i in caught)
    pixel_auroc = None
    if any(m is not None for m in masks):
        truth = np.stack([m if m is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool) for m in masks])
        pixel_auroc = float(roc_auc_score(truth.ravel(), maps.ravel()))
    image_auroc = float(roc_auc_score(is_defect, scores))
    tp, fp = int((flagged & is_defect).sum()), int((flagged & ~is_defect).sum())
    fn, tn = int((~flagged & is_defect).sum()), int((~flagged & ~is_defect).sum())
    kinds = ["good"] + sorted({s.kind for s in test} - {"good"})
    per_kind = [{"kind": k, "count": int(sum(s.kind == k for s in test)),
                 "flagged": int(sum(f for s, f in zip(test, flagged) if s.kind == k))} for k in kinds]
    metrics = {
        "image_auroc": image_auroc, "pixel_auroc": pixel_auroc,
        "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "per_kind": per_kind, "ms_per_image": ms_per_image, "fit_seconds": fit_seconds,
        "located": int(located_count), "located_of": len(caught),
    }
    print(f"    image AUROC {image_auroc:.3f}" + (f", pixel AUROC {pixel_auroc:.3f}" if pixel_auroc else "")
          + f"; caught {tp}/{tp + fn} defects, {fp}/{fp + tn} good parts wrongly flagged; "
          f"{ms_per_image:.0f} ms per image")
    if caught:
        print(f"    OpenCV boxes: the largest box is on the real defect in {located_count}/{len(caught)} caught defects")
    for r in per_kind:
        print(f"      {r['kind']:<20} flagged {r['flagged']}/{r['count']}")

    fig_dir = REPORTS_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    def add(file, caption):
        figures.append([file, caption])
        return fig_dir / file

    first_of_kind = [next(i for i, s in enumerate(test) if s.kind == k) for k in kinds][:8]
    plots.examples([{"image": view(test[i].path, crop, align), "heat": maps[i], "mask": masks[i],
                     "kind": test[i].kind, "score": float(scores[i])} for i in first_of_kind],
                   threshold, f"{name}: first test image of each type",
                   add("examples.png", "Heatmaps on test images"))
    plots.score_distribution(scores, is_defect, cal_scores, threshold,
                             add("score_distribution.png", "Scores of good vs. defective parts"))
    plots.roc(scores, is_defect, image_auroc, add("roc_curve.png", "ROC curve"))
    plots.per_kind(per_kind, add("per_type.png", "Result per defect type"))

    samples = [{"path": stored(test[i].path), "kind": test[i].kind}
               for k in kinds for i in [j for j, s in enumerate(test) if s.kind == k][:2]]
    meta = {
        "name": name, "source": source, "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "threshold": threshold, "learned_from": len(learn), "calibration_images": len(calibration),
        "test_images": len(test), "memory_bank": info["memory_bank"], "patches_seen": info["patches"],
        "metrics": metrics, "figures": figures, "samples": samples,
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }
    model.save(ARTIFACTS_DIR / name, meta)
    write_results_md(meta | {"crop": crop}, fig_dir / "RESULTS.md")
    return meta


def write_results_md(meta: dict, path: Path):
    m = meta["metrics"]
    lines = [
        f"# Results: {meta['name']}", "",
        f"Generated by `python -m src.train` on {meta['created']}. Source: {meta['source']}.", "",
        f"Learned from **{meta['learned_from']} good images** only "
        f"(memory bank of {meta['memory_bank']:,} patches out of {meta['patches_seen']:,}); "
        f"threshold set on {meta['calibration_images']} other good images; "
        f"tested on {meta['test_images']} images.", "",
        "| Metric | Value | Meaning |", "|---|---|---|",
        f"| Image AUROC | {m['image_auroc']:.3f} | does a defective part score higher than a good one? 1 = always |",
    ]
    if m["pixel_auroc"] is not None:
        lines.append(f"| Pixel AUROC | {m['pixel_auroc']:.3f} | do defect pixels score higher than normal pixels? |")
    lines += [
        f"| Defects caught | {m['true_positives']} / {m['true_positives'] + m['false_negatives']} "
        f"({m['recall']:.1%}) | at the threshold |",
        f"| Good parts wrongly flagged | {m['false_positives']} / {m['false_positives'] + m['true_negatives']} "
        f"({m['false_alarm_rate']:.1%}) | at the threshold |",
        f"| Speed | {m['ms_per_image']:.0f} ms per image | CPU, {platform.processor() or platform.machine()} |",
        "", "| Type | Flagged as defect |", "|---|---|",
    ]
    lines += [f"| {r['kind']} | {r['flagged']} / {r['count']} |" for r in m["per_kind"]]
    lines += ["", "## Figures", ""] + [f"### {c}\n\n![{c}]({f})\n" for f, c in meta["figures"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary():
    """reports/RESULTS.md and reports/summary.png: the public MVTec AD models side by side.

    Models trained on your own folder are left out on purpose: this file is meant to be shareable.
    The demo's Results tab still shows them on this laptop.
    """
    rows = []
    for meta_path in sorted(ARTIFACTS_DIR.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        m = meta["metrics"]
        rows.append({"name": meta["name"], "source": meta["source"], "image_auroc": m["image_auroc"],
                     "pixel_auroc": m["pixel_auroc"], "recall": m["recall"],
                     "false_alarm_rate": m["false_alarm_rate"], "ms": m["ms_per_image"]})
    rows = [r for r in rows if r["source"].startswith("MVTec")]
    if rows:
        plots.summary(rows, REPORTS_DIR / "summary.png")
    lines = ["# Results", "", "| Model | Data | Image AUROC | Pixel AUROC | Defects caught | Good parts flagged | ms / image |",
             "|---|---|---|---|---|---|---|"]
    def pixel(r):
        return "-" if r["pixel_auroc"] is None else f"{r['pixel_auroc']:.3f}"

    lines += [f"| [{r['name']}]({r['name']}/RESULTS.md) | {r['source']} | {r['image_auroc']:.3f} | {pixel(r)} | "
              f"{r['recall']:.1%} | {r['false_alarm_rate']:.1%} | {r['ms']:.0f} |" for r in rows]
    if rows:
        lines += ["", "![summary](summary.png)"]
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(SEED)
    print("Loading the pretrained Wide ResNet-50 (downloads 132 MB the first time) ...")
    extractor = FeatureExtractor()
    if args.data_dir:
        root = Path(args.data_dir)
        jobs = [(args.name or root.name.lower().replace(" ", "_"), root, f"your folder {root}")]
    else:
        jobs = [(c, download_mvtec(c), "MVTec AD (CC BY-NC-SA 4.0)") for c in args.categories]
    for name, root, source in jobs:
        run(name, root, source, not args.no_crop, args.coreset, args.max_train, args.calibration, extractor,
            align=name in args.align)
    write_summary()
    print(f"\nDone. Reports in {REPORTS_DIR}. Start the demo with: python app.py")


if __name__ == "__main__":
    main()
