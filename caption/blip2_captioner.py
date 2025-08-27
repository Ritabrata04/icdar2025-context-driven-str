"""
caption/blip2_captioner.py

BLIP-2 captioner producing medium-length contextual descriptions with a prompt
that emphasizes text-bearing objects and layout cues. Tuned for ~40–80 tokens.

If transformers or the specified BLIP-2 repo is not available, this module
raises a clear RuntimeError with installation hints.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


DEFAULT_PROMPT = (
    "Describe this image comprehensively, focusing on all visible text elements. "
    "Mention the environment, text-bearing objects, positions, and how the text "
    "relates to the scene context."
)


class BLIP2Captioner:
    """
    BLIP2Captioner(repo_id, revision, device, max_len, min_len)

    Example:
        cap = BLIP2Captioner('Salesforce/blip2-flan-t5-xl', device='cuda')
        caption = cap(img_rgb_np)
    """

    def __init__(
        self,
        repo_id: str = "Salesforce/blip2-flan-t5-xl",
        revision: Optional[str] = None,
        device: str = "cuda",
        max_len: int = 80,
        min_len: int = 40,
    ) -> None:
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except Exception as e:
            raise RuntimeError(
                "transformers not available. Please `pip install transformers accelerate`."
            ) from e

        self.device = device
        self.repo_id = repo_id
        self.revision = revision
        self.max_len = int(max_len)
        self.min_len = int(min_len)

        self._AutoProcessor = AutoProcessor
        self._AutoModelForVision2Seq = AutoModelForVision2Seq

        # Lazy init of heavy weights to allow construction in cold start if desired
        self._proc = None
        self._model = None

    def _lazy_init(self) -> None:
        if self._proc is None or self._model is None:
            self._proc = self._AutoProcessor.from_pretrained(self.repo_id, revision=self.revision)
            self._model = self._AutoModelForVision2Seq.from_pretrained(self.repo_id, revision=self.revision).to(self.device)
            self._model.eval()

    def __call__(self, img_rgb_np: np.ndarray, prompt: str = DEFAULT_PROMPT) -> str:
        import torch
        from PIL import Image

        if img_rgb_np.ndim != 3 or img_rgb_np.shape[2] != 3:
            raise ValueError("BLIP2Captioner expects an RGB array H×W×3.")

        self._lazy_init()
        pil = Image.fromarray(img_rgb_np)

        with torch.inference_mode():
            inputs = self._proc(images=pil, text=prompt, return_tensors="pt").to(self.device)
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_len,
                min_new_tokens=self.min_len,
                length_penalty=1.0,
                num_beams=3,
            )
            text = self._proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        return text
