"""
models/segmenter.py

A thin, reproducible wrapper around AdvancedUNet that:
  • Resizes RGB input to a fixed size (default 256×256),
  • Runs a forward pass (eval mode, no grad),
  • Resamples the probability map back to the original image resolution,
  • Thresholds to obtain a binary mask.

Design choices:
  • Deterministic: uses torch.inference_mode() and avoids random ops.
  • Standalone: includes `resize_with_record` locally (no external utils).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    # If torchvision is available, we can use it for better resizing; else fallback to torch.
    import torchvision.transforms.functional as TF
    _HAS_TV = True
except Exception:
    _HAS_TV = False

from .unet_adv import AdvancedUNet


@dataclass
class SegmenterConfig:
    """Lightweight configuration holder for the segmenter."""
    input_size: Tuple[int, int] = (256, 256)
    binarize_thresh: float = 0.5
    device: str = "cuda"
    ckpt: str | None = None


def _resize_with_record(img_rgb_np: np.ndarray, size: Tuple[int, int]) -> Tuple[torch.Tensor, dict]:
    """
    Resize an H×W×3 uint8 RGB image to `size` and return:
      -  CHW float32 tensor in [0,1],
      -  metadata dict with original H,W for back-projection.

    We do NOT pad/letterbox here to keep a simple 1:1 bilinear mapping.
    """
    if img_rgb_np.ndim != 3 or img_rgb_np.shape[2] != 3:
        raise ValueError("Expected RGB image of shape (H,W,3) uint8.")

    H0, W0 = img_rgb_np.shape[:2]
    Ht, Wt = size
    ten = torch.from_numpy(img_rgb_np).permute(2, 0, 1).float() / 255.0  # 3×H×W

    if _HAS_TV:
        ten = TF.resize(ten, [Ht, Wt], antialias=True)
    else:
        ten = ten.unsqueeze(0)
        ten = F.interpolate(ten, size=(Ht, Wt), mode="bilinear", align_corners=False)
        ten = ten.squeeze(0)

    meta = {"orig_hw": (H0, W0), "target_hw": (Ht, Wt)}
    return ten, meta


class Segmenter:
    """
    Segmenter(ckpt, device, thr) → callable wrapper over AdvancedUNet.

    Usage:
        seg = Segmenter(ckpt="...pth", device="cuda", thr=0.5)
        out = seg(img_rgb_np)  # dict with keys: 'mask', 'prob'
    """

    def __init__(self, ckpt: str | None, device: str = "cuda", thr: float = 0.5,
                 input_size: Tuple[int, int] = (256, 256)) -> None:
        self.model = AdvancedUNet(num_classes=1).to(device)
        if ckpt:
            sd = torch.load(ckpt, map_location=device)
            # Accept both {'state_dict':...} or raw state dict
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.device = device
        self.thr = float(thr)
        self.input_size = input_size

    @torch.inference_mode()
    def __call__(self, img_rgb_np: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference and return 'prob' (float32 H×W) and 'mask' (uint8 H×W)."""
        x, meta = _resize_with_record(img_rgb_np, size=self.input_size)
        x = x.unsqueeze(0).to(self.device)  # 1×3×Ht×Wt

        logits = self.model(x)              # expect 1×1×Ht×Wt
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise RuntimeError("AdvancedUNet must return shape [B,1,H,W].")
        prob_t = logits.sigmoid()

        # Resize back to original resolution
        H0, W0 = meta["orig_hw"]
        prob_t = F.interpolate(prob_t, size=(H0, W0), mode="bilinear", align_corners=False)

        prob = prob_t.squeeze().detach().cpu().numpy().astype(np.float32)    # H×W
        mask = (prob >= self.thr).astype(np.uint8)                            # H×W
        return {"prob": prob, "mask": mask}
