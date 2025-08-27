"""
models/fallback_deepsolo.py

Optional heavy recognizer fallback hook.
If a proper DeepSolo/MMOCR environment is present, this module can call it.
Otherwise it returns an empty string with a clear message.

Replace the stub `run_deepsolo_once` with your actual integration as needed.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import warnings


def run_deepsolo_once(img_rgb_np: np.ndarray, config_path: Optional[str] = None, ckpt_path: Optional[str] = None) -> str:
    """
    Placeholder for DeepSolo/MMOCR inference.

    Args:
        img_rgb_np: H×W×3 RGB image.
        config_path: path to DeepSolo config (optional).
        ckpt_path: path to DeepSolo checkpoint (optional).

    Returns:
        Recognized text string (best effort); empty string if not available.
    """
    try:
        # Example (pseudo-code):
        # from mmocr.apis import TextRecInferencer
        # infer = TextRecInferencer(model=config_path or 'deepsolo_config.py', weights=ckpt_path or 'deepsolo.pth', device='cuda')
        # result = infer(img_rgb_np, batch_size=1)
        # text = result['predictions'][0]['text']
        # return text
        raise NotImplementedError
    except Exception:
        warnings.warn("DeepSolo/MMOCR not available; returning empty string.")
        return ""


def maybe_deepsolo(img_rgb_np: np.ndarray, config_path: Optional[str] = None, ckpt_path: Optional[str] = None) -> str:
    """
    Public hook used by the pipeline. Safe to call even if DeepSolo isn't installed.
    """
    return run_deepsolo_once(img_rgb_np, config_path=config_path, ckpt_path=ckpt_path)
