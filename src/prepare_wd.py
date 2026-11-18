"""Copy a sample of a folder of good and defective images into the layout src.train expects.

    python -m src.prepare_wd --source "path/to/image folder"
    python -m src.train --data-dir data/wd_abs --name wd_abs --no-crop

Expected source:
    Good Sample/<...>/*.bmp, *.jpg       good parts
    Defect Sample/<defect type>/*.bmp    one folder per defect type

The copy goes to data/wd_abs/, which git ignores. Private data stays on this laptop:
never push it or upload it.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from .data import list_images
from .paths import DATA_DIR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="the '1_ABS image' folder")
    ap.add_argument("--out", default=str(DATA_DIR / "wd_abs"))
    ap.add_argument("--train", type=int, default=300, help="good images to learn from")
    ap.add_argument("--test-good", type=int, default=100, help="good images to test on")
    ap.add_argument("--per-defect", type=int, default=50, help="defective images per type to test on")
    args = ap.parse_args(argv)

    source, out = Path(args.source), Path(args.out)
    good_dirs = [d for d in [source, *source.rglob("*")] if d.is_dir() and d.name.lower().startswith("good")]
    good = sorted({p for d in good_dirs for p in list_images(d)})
    defect_root = next((d for d in source.iterdir() if d.is_dir() and d.name.lower().startswith("defect")), None)
    if not good or defect_root is None:
        raise SystemExit(f"Expected 'Good Sample/...' and 'Defect Sample/<type>/' folders in {source}")
    if len(good) < args.train + args.test_good:
        raise SystemExit(f"Only {len(good)} good images; lower --train/--test-good")

    rng = random.Random(0)
    rng.shuffle(good)
    plan = {"train/good": good[:args.train], "test/good": good[args.train:args.train + args.test_good]}
    for kind_dir in sorted(d for d in defect_root.iterdir() if d.is_dir()):
        images = list_images(kind_dir)
        if images:
            plan[f"test/{kind_dir.name.lower().replace(' ', '_')}"] = rng.sample(images, min(args.per_defect, len(images)))

    if out.exists():
        shutil.rmtree(out)
    for folder, files in plan.items():
        target = out / folder
        target.mkdir(parents=True)
        for i, path in enumerate(files):
            shutil.copy2(path, target / f"{i:04d}_{path.name}")
        print(f"  {folder:<28} {len(files):>4} images")
    print(f"\nReady: {out}\nNext:  python -m src.train --data-dir \"{out}\" --name wd_abs --no-crop")


if __name__ == "__main__":
    main()
