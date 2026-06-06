"""
postflight/lbp_descriptor.py — Local Binary Pattern texture descriptor.
Uses uniform LBP (radius=3, n_points=24) → 26-bin histogram.
Pure CPU — fast (~0.8ms per frame).
"""
import cv2
import numpy as np
import logging

try:
    from skimage.feature import local_binary_pattern
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False
    logging.getLogger(__name__).warning(
        "skimage not found — LBP falling back to manual implementation. "
        "pip install scikit-image for best results."
    )

log = logging.getLogger(__name__)

_RADIUS   = 3
_N_POINTS = 24
_N_BINS   = _N_POINTS + 2   # uniform LBP


def get_lbp_histogram(bgr_128: np.ndarray) -> np.ndarray:
    """
    Returns normalized 26-bin float32 histogram.
    Input: (128, 128, 3) uint8 BGR.
    """
    gray = cv2.cvtColor(bgr_128, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if _SKIMAGE:
        lbp = local_binary_pattern(gray, _N_POINTS, _RADIUS, method="uniform")
    else:
        lbp = _manual_lbp(gray)

    hist, _ = np.histogram(lbp.ravel(), bins=_N_BINS,
                           range=(0, _N_BINS), density=False)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def chi2_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi-squared distance between two histograms. Lower = more similar."""
    eps = 1e-10
    return float(0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps)))


def lbp_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Returns similarity in [0, 1]. 1 = identical."""
    d = chi2_distance(h1, h2)
    return float(1.0 / (1.0 + d))


def _manual_lbp(gray: np.ndarray) -> np.ndarray:
    """Fallback: basic 3x3 LBP without skimage."""
    h, w = gray.shape
    lbp = np.zeros_like(gray)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i, (dy, dx) in enumerate(neighbors):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp += (gray >= shifted).astype(np.uint8) * (1 << i)
    return lbp
