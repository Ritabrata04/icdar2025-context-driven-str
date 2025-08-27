"""
localize/mask_to_blocks.py

Connected-components based block localization (Algorithm 1 style):
  • Finds external contours on a binary mask,
  • Filters by area threshold A_min,
  • Applies p-pixel padding,
  • Sorts by area (desc),
  • Returns top-n (nmax) axis-aligned boxes.

All coordinates are returned in the ORIGINAL image space (same as mask).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class BlockParam:
    """Parameters for converting a mask to rectangular text blocks."""
    min_area: int = 150   # Amin
    pad: int = 6          # p
    nmax: int = 10        # max number of blocks


def _clip_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """Clip coordinates to valid image bounds."""
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    return x1, y1, x2, y2


def mask_to_blocks(mask: np.ndarray, params: BlockParam) -> List[Tuple[int, int, int, int]]:
    """
    Args:
        mask: H×W binary mask (uint8 or bool) in ORIGINAL size.
        params: BlockParam with thresholds and limits.

    Returns:
        A list of up to `nmax` bounding boxes (xmin, ymin, xmax, ymax), area-desc sorted.
    """
    if mask.ndim != 2:
        raise ValueError("mask_to_blocks expects a single-channel mask (H×W).")

    m = (mask > 0).astype(np.uint8)
    H, W = m.shape[:2]
    # Find external contours; CHAIN_APPROX_SIMPLE is sufficient for rectangles
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < params.min_area:
            continue
        x1 = x - params.pad
        y1 = y - params.pad
        x2 = x + w + params.pad
        y2 = y + h + params.pad
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, W, H)
        boxes.append((x1, y1, x2, y2))

    # Sort by area (descending) and take top-n
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[: params.nmax]
