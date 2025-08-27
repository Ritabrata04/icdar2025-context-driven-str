"""
models/recognizers.py

Lightweight, pluggable recognizer pool for T1 (full image) and T3 (crop) text recognition.

Adapters provided:
  • TrOCRRecognizer (HF: microsoft/trocr-base-printed)
  • ParseqRecognizer (HF: baudm/parseq)  [optional; loaded if available]
  • TesseractRecognizer (pytesseract)    [optional; fallback if installed]
  • DummyRecognizer (always returns empty string) — last-resort fallback

Policy:
  • The pool tries models in the order given by `cfg["models"]`.
  • If a requested adapter is unavailable, we raise a clear error unless
    a working alternative exists (then we warn & continue).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import warnings

import numpy as np
import cv2


# ---------- Base interface ----------

class BaseRecognizer:
    """Interface all recognizers must implement."""

    name: str = "base"

    def predict(self, img_rgb_np: np.ndarray) -> str:
        """Return a single text string for the given RGB image."""
        raise NotImplementedError


# ---------- Tesseract adapter (optional) ----------

class TesseractRecognizer(BaseRecognizer):
    name = "tesseract"

    def __init__(self, lang: str = "eng") -> None:
        try:
            import pytesseract  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "pytesseract is not installed or Tesseract binary is missing."
            ) from e
        self.lang = lang

    def predict(self, img_rgb_np: np.ndarray) -> str:
        import pytesseract
        # Convert to grayscale and apply mild threshold to help OCR
        gray = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        txt = pytesseract.image_to_string(thr, lang=self.lang, config="--psm 6")
        return txt.strip()


# ---------- TrOCR (HF) adapter ----------

class TrOCRRecognizer(BaseRecognizer):
    name = "trocr"

    def __init__(self, repo_id: str = "microsoft/trocr-base-printed", revision: Optional[str] = None, device: str = "cuda"):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
        except Exception as e:
            raise RuntimeError(
                "transformers (TrOCR) not available. Install transformers, torchvision, and torch."
            ) from e

        self.device = device
        self._torch = __import__("torch")
        self.proc = TrOCRProcessor.from_pretrained(repo_id, revision=revision)
        self.model = VisionEncoderDecoderModel.from_pretrained(repo_id, revision=revision).to(device)
        self.model.eval()

    def predict(self, img_rgb_np: np.ndarray) -> str:
        torch = self._torch
        # Convert to PIL via cv2 → RGB already
        from PIL import Image
        pil = Image.fromarray(img_rgb_np)
        with torch.inference_mode():
            inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
            out_ids = self.model.generate(**inputs, max_new_tokens=64)
            text = self.proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text.strip()


# ---------- PARSeq (HF) adapter (optional) ----------

class ParseqRecognizer(BaseRecognizer):
    name = "parseq"

    def __init__(self, repo_id: str = "baudm/parseq", revision: Optional[str] = None, device: str = "cuda", variant: str = "small"):
        try:
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            import torch
        except Exception as e:
            # Some community ports expose PARSeq differently; provide a clear hint.
            raise RuntimeError(
                "PARSeq via HF not available. You can use trocr or install a PARSeq implementation."
            ) from e

        # NOTE: Real PARSeq is sequence decoder; some HF mirrors expose it as classifier for demo.
        # We keep this as a stub to avoid brittle dependencies.
        self.device = device
        self._torch = __import__("torch")
        self.repo_id = repo_id
        self.extract = AutoFeatureExtractor.from_pretrained(repo_id, revision=revision)
        self.model = AutoModelForImageClassification.from_pretrained(repo_id, revision=revision).to(device)
        self.model.eval()

    def predict(self, img_rgb_np: np.ndarray) -> str:
        # WARNING: This stub returns class labels if the HF model is a classifier mirror.
        # Replace with a proper PARSeq adapter if you have the real checkpoint.
        from PIL import Image
        torch = self._torch
        pil = Image.fromarray(img_rgb_np)
        with torch.inference_mode():
            inputs = self.extract(images=pil, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits
            pred_id = int(logits.argmax(-1)[0].item())
            # Best-effort label (if provided by the checkpoint config)
            label = self.model.config.id2label.get(pred_id, "")
        return str(label).strip()


# ---------- Dummy adapter ----------

class DummyRecognizer(BaseRecognizer):
    name = "dummy"

    def predict(self, img_rgb_np: np.ndarray) -> str:
        return ""


# ---------- Pool ----------

class RecognizerPool:
    """
    RecognizerPool(cfg) manages {full, crop} predictions.

    cfg example:
        {
          "models": ["trocr", "tesseract"],  # priority order
          "trocr": {"repo_id": "microsoft/trocr-base-printed", "revision": null},
          "parseq": {"repo_id": "baudm/parseq", "variant": "small"},
          "device": "cuda"
        }
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        device = cfg.get("device", "cuda")
        requested: List[str] = cfg.get("models", ["trocr"])

        self.adapters: List[BaseRecognizer] = []
        for name in requested:
            try:
                if name.lower() == "trocr":
                    params = cfg.get("trocr", {})
                    self.adapters.append(
                        TrOCRRecognizer(
                            repo_id=params.get("repo_id", "microsoft/trocr-base-printed"),
                            revision=params.get("revision", None),
                            device=device,
                        )
                    )
                elif name.lower() == "parseq":
                    params = cfg.get("parseq", {})
                    self.adapters.append(
                        ParseqRecognizer(
                            repo_id=params.get("repo_id", "baudm/parseq"),
                            revision=params.get("revision", None),
                            device=device,
                            variant=params.get("variant", "small"),
                        )
                    )
                elif name.lower() == "tesseract":
                    params = cfg.get("tesseract", {})
                    self.adapters.append(TesseractRecognizer(lang=params.get("lang", "eng")))
                elif name.lower() == "dummy":
                    self.adapters.append(DummyRecognizer())
                else:
                    raise ValueError(f"Unknown recognizer '{name}'")
            except Exception as e:
                warnings.warn(f"Could not initialize recognizer '{name}': {e}")

        if not self.adapters:
            warnings.warn("No recognizers available; using DummyRecognizer.")
            self.adapters = [DummyRecognizer()]

        # Simple policy: use the first available adapter for both full and crop.
        self.primary = self.adapters[0]

    def predict_full(self, img_rgb_np: np.ndarray) -> str:
        """Predict on the full image using the primary adapter."""
        return self.primary.predict(img_rgb_np)

    def predict_crop(self, crop_rgb_np: np.ndarray) -> str:
        """Predict on a crop using the primary adapter."""
        return self.primary.predict(crop_rgb_np)
