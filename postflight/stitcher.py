"""
postflight/stitcher.py — Sequential homography-based mosaic stitching.
Uses SIFT + BFMatcher + RANSAC (switched from ORB — ORB's binary
descriptors were too weak on low-texture red-brown gravel soil,
producing <12 good matches per frame and falling back to an
unaligned center-blend that looked like a smeared double-exposure).
SIFT is slower but far more robust on repetitive/low-contrast terrain,
which is fine since stitching is post-flight only, not real-time.

If a frame still can't be aligned (insufficient matches even with
SIFT), it is now SKIPPED rather than blindly overlaid — an honest
gap in the mosaic is more useful for debugging than a falsely
"complete" but garbled image.

Processes frames in batches of 20 to stay within Jetson RAM limits.
Output mosaic is INTERNAL ONLY (visualization + spatial reasoning).
Coordinate source is always nav/pose metadata, not pixel positions.
"""
import cv2
import os
import json
import numpy as np
import logging
import gc
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)

STITCH_W, STITCH_H = 960, 540    # working resolution for stitching
BATCH_SIZE          = 20
MIN_MATCHES         = 12
RANSAC_THRESH       = 4.0
RATIO_TEST          = 0.72


class Stitcher:

    def __init__(self, mosaic_dir: str = "data/mosaic"):
        self._mosaic_dir = mosaic_dir
        os.makedirs(mosaic_dir, exist_ok=True)
        self._sift = cv2.SIFT_create(nfeatures=2000)
        self._bf   = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self._transforms: List[dict] = []

    def stitch(self, frame_paths: List[str], sortie_id: str) -> Optional[str]:
        """
        Stitch all frames. Returns path to output mosaic, or None on failure.
        """
        if len(frame_paths) < 2:
            log.warning("Stitcher: not enough frames to stitch")
            return None

        log.info(f"Stitcher: stitching {len(frame_paths)} frames in "
                 f"batches of {BATCH_SIZE}")

        # Process in batches → produce batch mosaics → stitch batch mosaics
        batches = [frame_paths[i:i+BATCH_SIZE]
                   for i in range(0, len(frame_paths), BATCH_SIZE)]

        batch_mosaics = []
        for bi, batch in enumerate(batches):
            log.info(f"  Batch {bi+1}/{len(batches)} ({len(batch)} frames)")
            mosaic = self._stitch_batch(batch, bi)
            if mosaic is not None:
                batch_mosaics.append(mosaic)
            gc.collect()

        if not batch_mosaics:
            log.error("Stitcher: all batches failed")
            return None

        if len(batch_mosaics) == 1:
            final = batch_mosaics[0]
        else:
            log.info("Stitcher: merging batch mosaics...")
            final = self._stitch_batch_list(batch_mosaics)

        out_path = os.path.join(self._mosaic_dir, f"mosaic_{sortie_id}.jpg")
        cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Save transform chain
        tx_path = os.path.join(self._mosaic_dir, f"transforms_{sortie_id}.json")
        with open(tx_path, "w") as f:
            json.dump({"sortie_id": sortie_id,
                       "frames": len(frame_paths),
                       "transforms": self._transforms}, f, indent=2)

        log.info(f"Stitcher: mosaic saved → {out_path}")
        return out_path

    # ── Internal ─────────────────────────────────────────────────────────────

    def _stitch_batch(self, paths: List[str], batch_idx: int) -> Optional[np.ndarray]:
        frames = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                log.warning(f"  Cannot read: {p}")
                continue
            img = cv2.resize(img, (STITCH_W, STITCH_H))
            frames.append(img)
        if not frames:
            return None

        canvas = frames[0].copy()
        frames[0] = None  # free original ref now that canvas holds a copy
        skipped = 0

        for i in range(1, len(frames)):
            target = frames[i]
            H = self._compute_homography(canvas, target, batch_idx, i)
            if H is not None:
                canvas = self._warp_and_blend(canvas, target, H)
            else:
                # Previously: blind center-blend overlay (caused the
                # smeared double-exposure look on low-texture soil).
                # Now: skip this frame, keep the mosaic honest.
                skipped += 1
                log.debug(f"  Batch {batch_idx} frame {i}: insufficient "
                          f"matches, skipping (not blending)")
            frames[i] = None  # release this frame's memory, keep list length stable
            gc.collect()

        if skipped:
            log.warning(f"  Batch {batch_idx}: {skipped}/{len(frames)-1} "
                        f"frames skipped (could not align)")

        return canvas

    def _stitch_batch_list(self, mosaics: List[np.ndarray]) -> np.ndarray:
        result = mosaics[0]
        for m in mosaics[1:]:
            H = self._compute_homography(result, m, -1, -1)
            if H is not None:
                result = self._warp_and_blend(result, m, H)
            else:
                log.warning("  Batch-mosaic merge: insufficient matches, skipping")
        return result

    def _compute_homography(self, base: np.ndarray, target: np.ndarray,
                            batch_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        g1 = cv2.cvtColor(base,   cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self._sift.detectAndCompute(g1, None)
        kp2, des2 = self._sift.detectAndCompute(g2, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return None

        matches = self._bf.knnMatch(des1, des2, k=2)
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < RATIO_TEST * n.distance:
                    good.append(m)

        self._transforms.append({
            "batch": batch_idx, "frame": frame_idx,
            "good_matches": len(good),
            "method": "homography" if len(good) >= MIN_MATCHES else "fallback"
        })

        if len(good) < MIN_MATCHES:
            log.debug(f"  Homography: only {len(good)} matches — using fallback")
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, RANSAC_THRESH)

        if H is None or abs(np.linalg.det(H)) < 0.1:
            log.debug("  Degenerate homography — fallback")
            return None

        return H

    @staticmethod
    def _warp_and_blend(base: np.ndarray, target: np.ndarray,
                        H: np.ndarray) -> np.ndarray:
        h, w = base.shape[:2]
        warped = cv2.warpPerspective(target, H, (w, h))
        # Simple alpha blend in overlap region
        mask_b = (base   > 0).any(axis=2).astype(np.float32)
        mask_w = (warped > 0).any(axis=2).astype(np.float32)
        overlap = (mask_b * mask_w)[..., np.newaxis]
        alpha   = 0.5
        result  = base.astype(np.float32).copy()
        result[mask_w.astype(bool)] = (
            (1.0 - alpha * overlap[mask_w.astype(bool)]) *
            result[mask_w.astype(bool)] +
            alpha * overlap[mask_w.astype(bool)] *
            warped[mask_w.astype(bool)].astype(np.float32)
        )
        # Fill empty regions from warped
        empty = (mask_b == 0) & (mask_w == 1)
        result[empty] = warped[empty].astype(np.float32)
        return result.astype(np.uint8)