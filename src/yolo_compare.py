"""Supervised YOLO detector vs. PatchCore, scored on the same held-out MVTec AD images.

    python -m src.yolo_compare                 # boxes from masks, train YOLO11n on CPU, compare
    python -m src.yolo_compare --epochs 30     # quicker

PatchCore learns from good parts only. YOLO needs photos of defects with boxes drawn on them. To compare
the two fairly, every product's test images are split in half per defect type (seed 0): YOLO trains on
one half, using boxes that OpenCV draws around the ground-truth masks, and both models are then scored
on the other half, which neither has seen. Run `python -m src.train` first (it trains PatchCore).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .data import DEFAULT_CATEGORIES, download_mvtec, load_folder
from .patchcore import FeatureExtractor, PatchCore
from .paths import ARTIFACTS_DIR, DATA_DIR, REPORTS_DIR
from .regions import mask_boxes

SEED = 0
CONF = 0.25  # a YOLO box at or above this confidence means "defect"


def split(test: list, seed: int = SEED) -> tuple[list, list]:
    """Half of each defect type (and of the good parts) for YOLO to learn from, half held out."""
    rng = random.Random(seed)
    learn, held = [], []
    for kind in sorted({s.kind for s in test}):
        group = [s for s in test if s.kind == kind]
        rng.shuffle(group)
        learn += group[:len(group) // 2]
        held += group[len(group) // 2:]
    return learn, held


def build_dataset(categories: list[str], out) -> dict:
    """YOLO folder layout; one class, "defect"; good images get an empty label file (background)."""
    if out.exists():
        shutil.rmtree(out)
    held_out, boxes = {}, 0
    for cat in categories:
        _, test = load_folder(download_mvtec(cat))
        learn, held = split(test)
        held_out[cat] = held
        for part, samples in (("train", learn), ("val", held)):
            (out / "images" / part).mkdir(parents=True, exist_ok=True)
            (out / "labels" / part).mkdir(parents=True, exist_ok=True)
            for s in samples:
                stem = f"{cat}_{s.kind}_{s.path.stem}"
                image = Image.open(s.path).convert("RGB")
                w, h = image.size
                image.save(out / "images" / part / f"{stem}.jpg", quality=95)
                lines = []
                if s.mask:
                    mask = np.array(Image.open(s.mask).convert("L")) > 127
                    for x, y, bw, bh in mask_boxes(mask):
                        lines.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
                boxes += len(lines) if part == "train" else 0
                (out / "labels" / part / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "names:\n  0: defect\n", encoding="utf-8")
    print(f"YOLO data: {sum(1 for _ in (out / 'images' / 'train').iterdir())} training images "
          f"({boxes} boxes), {sum(len(v) for v in held_out.values())} held-out images")
    return held_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=512)
    args = ap.parse_args(argv)

    from ultralytics import YOLO  # imported here so the rest of the project does not need it

    data_dir, run_dir = DATA_DIR / "yolo", ARTIFACTS_DIR / "yolo"
    held_out = build_dataset(args.categories, data_dir)

    started = time.time()
    YOLO(str(run_dir / "yolo11n.pt") if (run_dir / "yolo11n.pt").exists() else "yolo11n.pt").train(
        data=str(data_dir / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=16, device="cpu",
        workers=0, seed=SEED, deterministic=True, project=str(run_dir), name="run", exist_ok=True,
        plots=False, verbose=False)
    train_minutes = (time.time() - started) / 60
    detector = YOLO(str(run_dir / "run" / "weights" / "best.pt"))
    box = detector.val(data=str(data_dir / "data.yaml"), split="val", imgsz=args.imgsz, device="cpu",
                       plots=False, verbose=False).box

    extractor = FeatureExtractor()
    rows, examples = [], []
    for cat, held in held_out.items():
        paths = [str(s.path) for s in held]
        is_defect = np.array([s.is_defect for s in held])
        preds = detector.predict(paths, imgsz=args.imgsz, conf=CONF, device="cpu", verbose=False)
        yolo = np.array([len(p.boxes) > 0 for p in preds])
        patchcore, meta = PatchCore.load(ARTIFACTS_DIR / cat, extractor)
        scores, _ = patchcore.predict(paths)
        pc = scores > meta["threshold"]
        rows.append({
            "product": cat, "defects": int(is_defect.sum()), "good": int((~is_defect).sum()),
            "yolo_caught": int((yolo & is_defect).sum()), "yolo_false_alarms": int((yolo & ~is_defect).sum()),
            "patchcore_caught": int((pc & is_defect).sum()), "patchcore_false_alarms": int((pc & ~is_defect).sum()),
            "yolo_ms": float(np.mean([p.speed["inference"] for p in preds])),
        })
        first = next(i for i, s in enumerate(held) if s.is_defect)
        examples.append((cat, held[first].kind, preds[first].plot()[:, :, ::-1]))

    out = REPORTS_DIR / "yolo"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(rows))
    for k, (key, label) in enumerate([("patchcore_caught", "PatchCore (good parts only)"),
                                       ("yolo_caught", "YOLO11n (defect boxes)")]):
        ax.bar(x + (k - 0.5) * 0.38, [r[key] / r["defects"] for r in rows], 0.38, label=label)
    ax.set_xticks(x, [r["product"].replace("_", " ") for r in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("held-out defects caught")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Same held-out images, two approaches")
    fig.tight_layout()
    fig.savefig(out / "caught.png", dpi=150)
    plt.close(fig)
    strip = np.hstack([cv2.resize(img, (360, 360)) for _, _, img in examples])
    Image.fromarray(strip).save(out / "yolo_examples.png")

    result = {"epochs": args.epochs, "imgsz": args.imgsz, "conf": CONF, "train_minutes": train_minutes,
              "map50": float(box.map50), "map50_95": float(box.map), "rows": rows,
              "examples": [f"{c} / {k}" for c, k, _ in examples]}
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# YOLO vs. PatchCore", "",
        "Each product's MVTec AD test images were split in half per defect type (seed 0). YOLO11n (COCO-pretrained,",
        f"fine-tuned on CPU for {args.epochs} epochs at {args.imgsz} px, {train_minutes:.0f} min) learned from one half, with boxes",
        "drawn by OpenCV around the ground-truth masks. PatchCore learned from good training images only and never",
        "saw a defect. Both are scored on the other half. A YOLO box at confidence >= "
        f"{CONF} counts as a defect; PatchCore uses its own threshold set on good parts.", "",
        "| Product | Held-out defects | YOLO caught | PatchCore caught | YOLO false alarms | PatchCore false alarms |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['product']} | {r['defects']} | {r['yolo_caught']} | {r['patchcore_caught']} | "
                     f"{r['yolo_false_alarms']}/{r['good']} | {r['patchcore_false_alarms']}/{r['good']} |")
    lines += ["", f"YOLO box quality on the held-out half (all products): mAP50 {box.map50:.3f}, mAP50-95 {box.map:.3f}.",
              "", "![defects caught](caught.png)", "", "![YOLO boxes on held-out defects](yolo_examples.png)", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
