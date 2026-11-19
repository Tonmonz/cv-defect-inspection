"""OpenCV on the heatmap: turn the hot areas into boxes you can point at and measure."""
from __future__ import annotations

import cv2
import numpy as np


def defect_regions(heat: np.ndarray, threshold: float, min_area: int = 20, keep: int = 5) -> list[dict]:
    """Boxes around the parts of a heatmap that are hotter than anything seen on good parts.

    heat: (H, W) heatmap from PatchCore. threshold: the model's alarm level (set on good parts only).
    Returns up to `keep` regions, largest first: x, y, w, h and area in heatmap pixels, and the
    region's peak score as a multiple of the threshold.
    """
    hot = (heat > threshold).astype(np.uint8) * 255
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop single-pixel specks
    contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area),
                        "peak": float(heat[y:y + h, x:x + w].max() / threshold)})
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions[:keep]


def located(regions: list[dict], mask: np.ndarray) -> bool:
    """True when the largest region overlaps the real defect (the ground-truth mask)."""
    if not regions or mask is None:
        return False
    r = regions[0]
    return bool(mask[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]].any())


def mask_boxes(mask: np.ndarray, min_area: int = 4) -> list[tuple[int, int, int, int]]:
    """Boxes (x, y, w, h) around each defect in a ground-truth mask, for training a detector."""
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
