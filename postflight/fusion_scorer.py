"""
postflight/fusion_scorer.py — Adaptive DINOv2+LBP+HSV fusion scoring.
Weights are feature-type-specific (from config).
Automatically adjusts weights if DINOv2 fallback is active.
"""
import numpy as np
import logging
from typing import Dict, Tuple

from postflight.lbp_descriptor import lbp_similarity
from postflight.hsv_descriptor import hsv_similarity

log = logging.getLogger(__name__)


class FusionScorer:

    def __init__(self, matching_cfg: dict):
        self._weights: Dict[str, dict] = matching_cfg.get("weights", {})
        self._thresholds = matching_cfg.get("thresholds", {
            "high": 0.78, "medium": 0.65, "low": 0.52
        })
        self._dino_fallback = False

    def set_dino_fallback(self, active: bool):
        """Call if DINOv2 fell back to MobileNetV2. Adjusts weights slightly."""
        self._dino_fallback = active
        if active:
            log.warning("FusionScorer: DINOv2 fallback — redistributing weights")

    def _get_weights(self, feature_type: str) -> Tuple[float, float, float]:
        """Returns (w_dino, w_lbp, w_hsv), adjusted for fallback if needed."""
        defaults = {"dino": 0.40, "lbp": 0.30, "hsv": 0.30}
        w = self._weights.get(feature_type, defaults)
        wd = float(w.get("dino", 0.40))
        wl = float(w.get("lbp",  0.30))
        wh = float(w.get("hsv",  0.30))

        if self._dino_fallback:
            # Reduce DINO weight by 0.10, redistribute equally to LBP + HSV
            wd = max(0.10, wd - 0.10)
            extra = 0.10 / 2.0
            wl += extra
            wh += extra

        # Normalize to sum to 1.0
        total = wd + wl + wh
        if total > 0:
            wd /= total; wl /= total; wh /= total

        return wd, wl, wh

    def score(self,
              query_dino: np.ndarray, query_lbp: np.ndarray, query_hsv: np.ndarray,
              ref_dino:   np.ndarray, ref_lbp:   np.ndarray, ref_hsv:   np.ndarray,
              feature_type: str) -> Tuple[float, dict]:
        """
        Returns (fusion_score: float, breakdown: dict).
        fusion_score in [0, 1].
        """
        # DINO: cosine similarity (vectors already L2-normalized)
        dino_sim = float(np.dot(query_dino, ref_dino))
        dino_sim = max(0.0, (dino_sim + 1.0) / 2.0)   # shift [-1,1] → [0,1]

        lbp_sim  = lbp_similarity(query_lbp, ref_lbp)
        hsv_sim  = hsv_similarity(query_hsv, ref_hsv)

        wd, wl, wh = self._get_weights(feature_type)
        fusion = wd * dino_sim + wl * lbp_sim + wh * hsv_sim

        return float(fusion), {
            "dino": round(dino_sim, 4),
            "lbp":  round(lbp_sim,  4),
            "hsv":  round(hsv_sim,  4),
            "weights": {"dino": round(wd, 3), "lbp": round(wl, 3), "hsv": round(wh, 3)},
        }

    def get_tier(self, score: float) -> str:
        th = self._thresholds
        if score >= th.get("high",   0.78): return "HIGH"
        if score >= th.get("medium", 0.65): return "MEDIUM"
        if score >= th.get("low",    0.52): return "LOW"
        return "REJECT"

    def is_valid_detection(self, score: float) -> bool:
        return score >= self._thresholds.get("low", 0.52)
