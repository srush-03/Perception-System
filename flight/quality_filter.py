"""
flight/quality_filter.py — Frame quality gate.
Every captured frame passes through here before keyframe selection.
Any single failed check causes immediate rejection.
"""
import cv2
import numpy as np
import logging
from typing import Tuple

log = logging.getLogger(__name__)


class QualityFilter:

    def __init__(self, cfg: dict):
        self.blur_min   = cfg.get("blur_laplacian_min",  80.0)
        self.v_max      = cfg.get("overexposure_v_max",  240)
        self.v_min      = cfg.get("underexposure_v_min", 30)
        self.entropy_min= cfg.get("entropy_min",         4.5)
        self.s_max      = cfg.get("solid_color_s_max",   8)

    def check(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Returns (passed: bool, reason: str).
        reason is 'ok' if passed, otherwise the rejection cause.
        """
        if frame is None or frame.size == 0:
            return False, "null_frame"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Blur check (CV_32F — avoids AVX2 bug with CV_64F) ──
        lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
        if lap_var < self.blur_min:
            return False, f"blur:{lap_var:.1f}<{self.blur_min}"

        # ── Exposure check via HSV V channel ──
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_mean = hsv[:, :, 2].mean()
        s_mean = hsv[:, :, 1].mean()

        if v_mean > self.v_max:
            return False, f"overexposed:V={v_mean:.1f}"
        if v_mean < self.v_min:
            return False, f"underexposed:V={v_mean:.1f}"

        # ── Solid color check ──
        if s_mean < self.s_max:
            return False, f"solid_color:S={s_mean:.1f}"

        # ── Shannon entropy ──
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob))
        if entropy < self.entropy_min:
            return False, f"low_entropy:{entropy:.2f}<{self.entropy_min}"

        return True, "ok"
