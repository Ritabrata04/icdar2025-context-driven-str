"""
I/O helpers for images and artifacts; kept dependency-light and explicit.

Functions:
- imread_rgb(path)         → H×W×3 RGB np.uint8 (raises on error)
- imwrite(path, bgr/gray)  → writes with parent creation
- ensure_dir(path)         → mkdir -p
- draw_boxes(img, boxes)   → quick visualization for debugging
- save_mask(mask, path)    → saves binary mask as PNG
- list_images(root)        → sorted list of image paths (common extensions)
"""

from __future__ import annotations
from typing import List, Tuple
import os, pathlib

import cv2
import numpy as np


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def imread_rgb(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"OpenCV could not read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def save_mask(mask: np.ndarray, path: str) -> None:
    """
    Saves a binary mask (H×W) as 8-bit PNG (0/255).
    """
    m = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    if m.max() == 1:
        m = m * 255
    imwrite(path, m)


def draw_boxes(img_rgb: np.ndarray, boxes: List[Tuple[int,int,int,int]], thickness: int = 2) -> np.ndarray:
    """
    Return a BGR copy with rectangle overlays; useful for quick debugging.
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), thickness)
    return bgr


def list_images(root: str) -> List[str]:
    out = []
    for p, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f.lower())[1] in _IMG_EXTS:
                out.append(os.path.join(p, f))
    return sorted(out)
