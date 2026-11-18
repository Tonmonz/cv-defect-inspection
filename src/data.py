"""Getting images: download MVTec AD categories, or use your own folder laid out the same way.

The layout (MVTec AD's, also used for your own data):

    <folder>/train/good/*.png              defect-free images to learn from
    <folder>/test/good/*.png               defect-free images to test on
    <folder>/test/<defect type>/*.png      defective images to test on
    <folder>/ground_truth/<defect type>/<image name>_mask.png   (optional) where the defect is
"""
from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .paths import MVTEC_DIR

# Official per-category links from https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
_BASE = "https://www.mydrive.ch/shares/"
MVTEC_URLS = {
    "bottle": "163640/9a565b9a02f45cd135e46b1c36e01ec5/download/420937370-1629958698/bottle.tar.xz",
    "cable": "163641/97fd0295152453e533cfb3db1d253a35/download/420937413-1629958794/cable.tar.xz",
    "capsule": "163642/236d4304eaf9526b350b18807789ce86/download/420937454-1629958872/capsule.tar.xz",
    "carpet": "163643/70e8bebbe9f2d2c55043311a4e67eb1d/download/420937484-1629959013/carpet.tar.xz",
    "grid": "163644/8c7fc7cba43ef5275c87ca750a6d0004/download/420937487-1629959044/grid.tar.xz",
    "hazelnut": "163645/8346f81c9e093d2ebbb3a73f98372217/download/420937545-1629959162/hazelnut.tar.xz",
    "leather": "163646/7e0627b5718d2902bb85d24cff7a51ce/download/420937607-1629959262/leather.tar.xz",
    "metal_nut": "163647/6d5a833cdee9035a45d5ce0921ecd24d/download/420937637-1629959294/metal_nut.tar.xz",
    "pill": "163648/f8c4ca0d325888f2ba128a9ed44708f5/download/420938129-1629960351/pill.tar.xz",
    "screw": "163649/cfd394b4e3a49013745fab9e2084988a/download/420938130-1629960389/screw.tar.xz",
    "tile": "163650/55ad2038770deff6fe6771bd784af8cc/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": "163651/f03d8421ec64b3e3bfa4a2ba2fbafb4f/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": "163652/619a88c3d5f3f57b46b00f44f04ad6e6/download/420938166-1629960554/transistor.tar.xz",
    "wood": "163653/937e0d65bc9c4497d253cb7313868a60/download/420938383-1629960649/wood.tar.xz",
    "zipper": "163654/7f6e6aefbf00a9e8a09cc0bbac236eb0/download/420938385-1629960680/zipper.tar.xz",
}
DEFAULT_CATEGORIES = ["bottle", "metal_nut", "screw"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    kind: str  # "good" or the defect type (the folder name)
    mask: Path | None = None  # where the defect is, if the dataset says

    @property
    def is_defect(self) -> bool:
        return self.kind != "good"


def _progress(blocks, block_size, total):
    if total > 0:
        done = min(blocks * block_size, total)
        print(f"\r      {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)


def download_mvtec(category: str) -> Path:
    """Download and unpack one MVTec AD category into data/mvtec/<category> (once)."""
    folder = MVTEC_DIR / category
    if (folder / "train" / "good").is_dir():
        return folder
    if category not in MVTEC_URLS:
        raise SystemExit(f"Unknown MVTec AD category '{category}'. Choose from: {', '.join(MVTEC_URLS)}")
    MVTEC_DIR.mkdir(parents=True, exist_ok=True)
    archive = MVTEC_DIR / f"{category}.tar.xz"
    print(f"      downloading MVTec AD '{category}' (CC BY-NC-SA 4.0) ...")
    urllib.request.urlretrieve(_BASE + MVTEC_URLS[category], archive, _progress)
    print("\n      unpacking ...")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(MVTEC_DIR, filter="data")
    archive.unlink()
    return folder


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if folder.is_dir() else []


def load_folder(root: Path) -> tuple[list[Path], list[Sample]]:
    """Training images (good only) and labelled test images of a folder in the layout above."""
    train = list_images(root / "train" / "good")
    if not train:
        raise SystemExit(f"No training images in {root / 'train' / 'good'}")
    test = []
    for kind_dir in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        for path in list_images(kind_dir):
            mask = root / "ground_truth" / kind_dir.name / f"{path.stem}_mask.png"
            test.append(Sample(path, kind_dir.name, mask if mask.exists() else None))
    if not any(s.is_defect for s in test) or not any(not s.is_defect for s in test):
        raise SystemExit(f"{root / 'test'} needs a 'good' folder and at least one defect folder")
    return train, test
