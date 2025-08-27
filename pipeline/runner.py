"""
End-to-end pipeline runner:
  1) Segment → mask (AdvancedUNet wrapper)
  2) Localize → block boxes via connected components
  3) Caption  → BLIP-2 medium-length contextual caption (T2)
  4) Recognize:
        - T1 on full image (primary recognizer)
        - T3 on crops: evaluate ALL crops, keep the one with max C3
  5) Score    → compute S1,S3,L1,L3,C1,C3 and choose T_final if confident
  6) Fallback → DeepSolo/MMOCR only if below τ (optional, pluggable)
  7) Save artifacts (mask, crops) and emit a structured RunOutputs

This runner is stateless and pure; it accepts a resolved OmegaConf `cfg` or a dict.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import cv2

try:
    from omegaconf import OmegaConf
    _HAS_OMEGA = True
except Exception:
    _HAS_OMEGA = False

from models.segmenter import Segmenter
from localize.mask_to_blocks import mask_to_blocks, BlockParam
from caption.blip2_captioner import BLIP2Captioner
from models.recognizers import RecognizerPool
from scoring.context_score import ContextScorer
from models.fallback_deepsolo import maybe_deepsolo
from pipeline.io_utils import save_mask


@dataclass
class CropInfo:
    box: tuple
    text: str
    S3: float
    L3: float
    C3: float


@dataclass
class RunOutputs:
    image_path: str
    mask: np.ndarray
    boxes: List[tuple]
    crops: List[CropInfo]
    T1: str
    T2: str
    best_T3: str
    S1: float
    L1: float
    best_C3: float
    best_S3: float
    best_L3: float
    C1: float
    Cmax: float
    Tfinal: str
    used_fallback: bool
    seconds: float

    def as_row(self) -> List[Any]:
        return [
            self.image_path, self.T1, self.best_T3, self.T2,
            self.S1, self.best_S3, self.L1, self.best_L3,
            self.C1, self.best_C3, self.Cmax, self.Tfinal,
            int(self.used_fallback), self.seconds,
        ]


class PipelineRunner:
    def __init__(self, cfg: Any):
        """
        cfg may be a dict or an OmegaConf (DictConfig). Expected keys:
            cfg.segmenter.ckpt, cfg.segmenter.binarize_thresh
            cfg.localize.min_area, cfg.localize.pad, cfg.localize.nmax
            cfg.caption.model, cfg.caption.max_len, cfg.caption.min_len
            cfg.recognition (dict): {"models": [...], "...": {...}}
            cfg.scoring: embedder, alpha, beta, tau
            cfg.fallback: use_deepsolo: bool
            cfg.device: "cuda" or "cpu"
        """
        self.cfg = cfg
        device = _get(cfg, "device", "cuda")

        # 1) segmentation
        self.seg = Segmenter(
            ckpt=_get(cfg, "segmenter.ckpt", None),
            device=device,
            thr=float(_get(cfg, "segmenter.binarize_thresh", 0.5)),
            input_size=tuple(_get(cfg, "segmenter.input_size", (256, 256))),
        )

        # 2) localization
        self.block_param = BlockParam(
            min_area=int(_get(cfg, "localize.min_area", 150)),
            pad=int(_get(cfg, "localize.pad", 6)),
            nmax=int(_get(cfg, "localize.nmax", 10)),
        )

        # 3) caption (T2)
        self.captioner = BLIP2Captioner(
            repo_id=_get(cfg, "caption.model", "Salesforce/blip2-flan-t5-xl"),
            revision=None,
            device=device,
            max_len=int(_get(cfg, "caption.max_len", 80)),
            min_len=int(_get(cfg, "caption.min_len", 40)),
        )

        # 4) recognizers (T1/T3)
        self.recogs = RecognizerPool(_get(cfg, "recognition", {}))

        # 5) scorer
        self.scorer = ContextScorer(
            embedder_repo=_get(cfg, "scoring.embedder", "sentence-transformers/all-mpnet-base-v2"),
            alpha=float(_get(cfg, "scoring.alpha", 0.6)),
            beta=float(_get(cfg, "scoring.beta", 0.4)),
            tau=float(_get(cfg, "scoring.tau", 0.8)),
            device=device,
        )

        # 6) fallback
        self.use_fallback = bool(_get(cfg, "fallback.use_deepsolo", True))
        self.only_if_below_tau = bool(_get(cfg, "fallback.only_if_below_tau", True))

    def run_image(self, img_rgb: np.ndarray, image_path: str = "") -> RunOutputs:
        """
        Run the full pipeline on a single RGB image.
        """
        import time
        t0 = time.time()

        # Segmentation
        seg = self.seg(img_rgb)
        mask = seg["mask"]

        # Localization → get blocks
        boxes = mask_to_blocks(mask, self.block_param)

        # Caption (T2)
        T2 = self.captioner(img_rgb)

        # T1: full-image recognition
        T1 = self.recogs.predict_full(img_rgb)

        # C1 (requires S1/L1 wrt T2)
        s_full = self.scorer.score(T1=T1, T3="", T2=T2)
        S1, L1, C1 = s_full["S1"], s_full["L1"], s_full["C1"]

        # T3: evaluate ALL crops and keep the best C3
        crops_info: List[CropInfo] = []
        best_T3 = ""
        best_C3 = -1.0
        best_S3 = 0.0
        best_L3 = 0.0

        for (x1, y1, x2, y2) in boxes:
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            T3 = self.recogs.predict_crop(crop)
            s = self.scorer.score(T1="", T3=T3, T2=T2)
            ci = CropInfo(box=(x1, y1, x2, y2), text=T3, S3=s["S3"], L3=s["L3"], C3=s["C3"])
            crops_info.append(ci)
            if ci.C3 > best_C3:
                best_C3, best_S3, best_L3, best_T3 = ci.C3, ci.S3, ci.L3, ci.text

        # Decision: combine T1 vs best_T3
        Cmax = max(C1, best_C3)
        used_fallback = False
        if Cmax >= self.scorer.tau:
            # Pick semantic tie-break: higher S gets priority
            if best_T3 and best_S3 > S1:
                Tfinal = best_T3
            else:
                Tfinal = T1
        else:
            # Below tau → optional fallback
            Tfinal = ""
            if self.use_fallback:
                Tfinal = maybe_deepsolo(img_rgb)
                used_fallback = True

        seconds = time.time() - t0
        return RunOutputs(
            image_path=image_path,
            mask=mask,
            boxes=boxes,
            crops=crops_info,
            T1=T1,
            T2=T2,
            best_T3=best_T3,
            S1=S1,
            L1=L1,
            best_C3=best_C3 if best_C3 >= 0 else 0.0,
            best_S3=best_S3,
            best_L3=best_L3,
            C1=C1,
            Cmax=Cmax if Cmax >= 0 else 0.0,
            Tfinal=Tfinal,
            used_fallback=used_fallback,
            seconds=seconds,
        )


def _get(cfg: Any, dotted: str, default: Any) -> Any:
    """
    Safe nested-get supporting both dict and OmegaConf.DictConfig with dotted paths.
    """
    if _HAS_OMEGA and isinstance(cfg, OmegaConf.__class__):
        return OmegaConf.select(cfg, dotted, default=default)
    # Fallback: naive dict traversal
    cur = cfg
    for key in dotted.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur
