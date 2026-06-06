"""
postflight/hsv_descriptor.py — HSV color histogram descriptor.
H: 36 bins, S: 32 bins, V: 32 bins → 100-dim L2-normalized vector.
Pure CPU — fast (~0.5ms per frame).
"""
import cv2
import numpy as np


def get_hsv_histogram(bgr_128: np.ndarray) -> np.ndarray:
    """
    Returns 100-dim L2-normalized float32 HSV histogram.
    Input: (128, 128, 3) uint8 BGR.
    """
    hsv = cv2.cvtColor(bgr_128, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    combined = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm > 1e-8:
        combined /= norm
    return combined


def bhattacharyya_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """OpenCV Bhattacharyya distance. Lower = more similar."""
    # Reshape as OpenCV expects (N, 1) float32
    a = h1.reshape(-1, 1).astype(np.float32)
    b = h2.reshape(-1, 1).astype(np.float32)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def hsv_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Returns similarity in [0, 1]. 1 = identical."""
    d = bhattacharyya_distance(h1, h2)
    return float(max(0.0, 1.0 - d))
