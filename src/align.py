"""OpenCV part alignment: turn every part to the same angle before the model looks at it.

Screws lie at any angle, so a good screw at an unusual angle looks "unusual" to a model that
learned from good screws. Aligning first removes that difference:
  1. Otsu threshold separates the part from the background (the part is the smaller side).
  2. The largest contour is the part; image moments give its centre and long-axis angle.
  3. Rotate so the long axis is horizontal and the centre sits in the middle, then flip
     180 degrees if needed so the heavier end (a screw's head) always points left.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _part_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() > 127:  # the part covers less of the picture than the background
        mask = 255 - mask
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def alignment(image: Image.Image) -> np.ndarray | None:
    """2 x 3 affine matrix that aligns the part, or None when no part is found."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = _part_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    part = max(contours, key=cv2.contourArea)
    m = cv2.moments(part)
    if m["m00"] < 0.002 * h * w:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    angle = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    matrix[0, 2] += w / 2 - cx
    matrix[1, 2] += h / 2 - cy
    turned = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
    columns = turned.sum(axis=0)
    if columns[w // 2:].sum() > columns[:w // 2].sum():  # heavier end on the right: turn it around
        flip = cv2.getRotationMatrix2D((w / 2, h / 2), 180, 1.0)
        matrix = flip @ np.vstack([matrix, [0, 0, 1]])
    return matrix


def warp(array: np.ndarray, matrix: np.ndarray, nearest: bool = False) -> np.ndarray:
    h, w = array.shape[:2]
    return cv2.warpAffine(array, matrix, (w, h), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT if nearest else cv2.BORDER_REPLICATE)


def align_image(image: Image.Image) -> tuple[Image.Image, np.ndarray | None]:
    matrix = alignment(image)
    if matrix is None:
        return image, None
    return Image.fromarray(warp(np.asarray(image.convert("RGB")), matrix)), matrix
