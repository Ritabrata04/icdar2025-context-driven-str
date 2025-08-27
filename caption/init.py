"""
Public API for `caption` package.
Exports:
- BLIP2Captioner
"""
from .blip2_captioner import BLIP2Captioner, DEFAULT_PROMPT

__all__ = ["BLIP2Captioner", "DEFAULT_PROMPT"]
