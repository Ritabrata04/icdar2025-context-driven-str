"""
Public API for `models` package.
Exports:
- AdvancedUNet        (from .unet_adv)  — your attention-aided UNet
- Segmenter           (from .segmenter) — wrapper that returns masks in original size
- RecognizerPool      (from .recognizers) — lightweight STR pool with adapters
- maybe_deepsolo      (from .fallback_deepsolo) — optional heavy fallback
"""
from .unet_adv import AdvancedUNet  # provided by you
from .segmenter import Segmenter
from .recognizers import RecognizerPool
from .fallback_deepsolo import maybe_deepsolo

__all__ = ["AdvancedUNet", "Segmenter", "RecognizerPool", "maybe_deepsolo"]
