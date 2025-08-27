"""
scoring/context_score.py

Implements the semantic + lexical context scoring scheme:

Inputs:
  • T1 : text predicted on full image
  • T3 : text predicted on top crop
  • T2 : BLIP-2 caption (medium length)

Scores:
  • S1 = cos_sim(emb(T1), emb(T2))
  • S3 = cos_sim(emb(T3), emb(T2))
  • L1 = Levenshtein-based similarity(T1, T2) in [0,1]
  • L3 = Levenshtein-based similarity(T3, T2) in [0,1]
  • C1 = α S1 + β L1
  • C3 = α S3 + β L3

Decision:
  • If max(C1, C3) ≥ τ then return T_final = argmax_S {T1, T3} (semantic tie-break).
  • Else return empty string and signal `confident=False` so a heavy fallback can run.

Dependencies:
  • sentence-transformers (all-mpnet-base-v2 recommended)
  • rapidfuzz (for Levenshtein similarity)

Reproducibility:
  • Uses deterministic encode() and pure cosine similarity from SentenceTransformers.util.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ScoreConfig:
    embedder_repo: str = "sentence-transformers/all-mpnet-base-v2"
    alpha: float = 0.6
    beta: float = 0.4
    tau: float = 0.8
    device: str = "cuda"


class ContextScorer:
    """Context scorer with MPNet embeddings + Levenshtein similarity."""

    def __init__(self, embedder_repo: str, alpha: float = 0.6, beta: float = 0.4, tau: float = 0.8, device: str = "cuda"):
        try:
            from sentence_transformers import SentenceTransformer, util
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers not available. Please `pip install sentence-transformers`."
            ) from e
        try:
            from rapidfuzz.fuzz import ratio as fuzz_ratio  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "rapidfuzz not available. Please `pip install rapidfuzz`."
            ) from e

        self._SentenceTransformer = SentenceTransformer
        self._util = util
        self._fuzz_ratio = __import__("rapidfuzz.fuzz", fromlist=["ratio"]).ratio

        self.model = SentenceTransformer(embedder_repo, device=device)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)

    def _cos(self, a, b) -> float:
        score = float(self._util.cos_sim(a, b))
        # Clamp for numerical stability
        return max(min(score, 1.0), -1.0)

    def _lev(self, a: str, b: str) -> float:
        # rapidfuzz returns [0,100]; normalize to [0,1]
        return float(self._fuzz_ratio(a, b)) / 100.0

    def score(self, T1: str, T3: str, T2: str) -> Dict[str, float | str | bool]:
        """
        Compute S/L/C scores and final decision.

        Returns:
            {
              "S1","S3","L1","L3","C1","C3",
              "Tfinal": str,
              "confident": bool
            }
        """
        # Guard empty inputs (embedder can still encode "", but it’s meaningless)
        T1 = T1 or ""
        T3 = T3 or ""
        T2 = T2 or ""

        # Encode (CPU/GPU handled by model.device)
        e1, e3, e2 = self.model.encode([T1, T3, T2], convert_to_tensor=True, normalize_embeddings=True)

        S1 = self._cos(e1, e2)
        S3 = self._cos(e3, e2)
        L1 = self._lev(T1, T2)
        L3 = self._lev(T3, T2)

        C1 = self.alpha * S1 + self.beta * L1
        C3 = self.alpha * S3 + self.beta * L3

        confident = max(C1, C3) >= self.tau
        if confident:
            # Tie-break by higher semantic alignment (S), not by C, to bias towards meaning.
            Tfinal = T1 if S1 >= S3 else T3
        else:
            Tfinal = ""

        return {
            "S1": float(S1),
            "S3": float(S3),
            "L1": float(L1),
            "L3": float(L3),
            "C1": float(C1),
            "C3": float(C3),
            "Tfinal": Tfinal,
            "confident": bool(confident),
        }
