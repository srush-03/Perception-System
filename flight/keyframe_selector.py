"""
flight/keyframe_selector.py — Decides whether a quality-passed frame
is a new keyframe or a redundant duplicate of the last accepted frame.

Methods used:
  1. Optical flow displacement (motion gating)
  2. Histogram Bhattacharyya distance
  3. SSIM similarity
  4. ORB feature richness
  5. Forced keyframe every N seconds (coverage guarantee)
"""
import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple

try:
    from skimage.metrics import structural_similarity as ssim
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

log = logging.getLogger(__name__)


class KeyframeSelector:

    def __init__(self, cfg: dict):
        self.motion_thresh   = cfg.get("motion_threshold_px",  15)
        self.hist_diff_min   = cfg.get("histogram_diff_min",   0.12)
        self.ssim_reject     = cfg.get("ssim_reject_above",    0.88)
        self.min_orb         = cfg.get("min_orb_features",     80)
        self.force_every_sec = cfg.get("force_every_sec",      3.0)

        self._prev_gray:  Optional[np.ndarray] = None
        self._prev_pts:   Optional[np.ndarray] = None
        self._prev_hist:  Optional[np.ndarray] = None
        self._last_kf_ts: float = 0.0
        self._orb = cv2.ORB_create(nfeatures=300)

    def is_keyframe(self, frame: np.ndarray,
                    timestamp: float) -> Tuple[bool, str]:
        """
        Returns (is_keyframe: bool, reason: str).
        First frame is always accepted.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (320, 240))

        # ── First frame ──────────────────────────────────────────────────────
        if self._prev_gray is None:
            self._accept(gray, gray_small, timestamp)
            return True, "first_frame"

        # ── Forced keyframe (coverage guarantee) ─────────────────────────────
        if (timestamp - self._last_kf_ts) >= self.force_every_sec:
            self._accept(gray, gray_small, timestamp)
            return True, "forced_interval"

        # ── Motion check via sparse optical flow ─────────────────────────────
        if self._prev_pts is not None and len(self._prev_pts) > 0:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray,
                self._prev_pts, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            good = status.ravel() == 1
            if good.sum() > 5:
                motion = np.linalg.norm(
                    new_pts[good] - self._prev_pts[good], axis=1
                ).mean()
                if motion < self.motion_thresh:
                    # small motion — check histogram difference too
                    curr_hist = self._compute_hist(gray_small)
                    if self._prev_hist is not None:
                        diff = cv2.compareHist(
                            self._prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA
                        )
                        if diff < self.hist_diff_min:
                            return False, f"redundant:motion={motion:.1f}px,hist={diff:.3f}"

        # ── SSIM check ───────────────────────────────────────────────────────
        if _SKIMAGE:
            prev_small = cv2.resize(self._prev_gray, (320, 240))
            sim = ssim(prev_small, gray_small, data_range=255)
            if sim > self.ssim_reject:
                return False, f"redundant:ssim={sim:.3f}"

        # ── Feature richness check ───────────────────────────────────────────
        kps = self._orb.detect(gray_small, None)
        if len(kps) < self.min_orb:
            return False, f"low_features:{len(kps)}<{self.min_orb}"

        self._accept(gray, gray_small, timestamp)
        return True, "new_content"

    def _accept(self, gray: np.ndarray, gray_small: np.ndarray,
                timestamp: float):
        # Sample feature points for next optical flow
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=10
        )
        self._prev_gray  = gray
        self._prev_pts   = pts
        self._prev_hist  = self._compute_hist(gray_small)
        self._last_kf_ts = timestamp

    @staticmethod
    def _compute_hist(gray_small: np.ndarray) -> np.ndarray:
        h = cv2.calcHist([gray_small], [0], None, [64], [0, 256])
        cv2.normalize(h, h)
        return h

    def reset(self):
        """Call between sorties to clear state."""
        self._prev_gray  = None
        self._prev_pts   = None
        self._prev_hist  = None
        self._last_kf_ts = 0.0
