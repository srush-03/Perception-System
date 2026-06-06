"""
postflight/matcher.py — Per-frame hybrid matching pipeline.
For each HD keyframe:
  1. Load HD from disk (lazy)
  2. Downsample to 128x128 in memory (NEVER saved)
  3. Extract DINOv2 + LBP + HSV
  4. Score against all reference embeddings (all feature types)
  5. Record detections → raw_detections.json
  6. Release frame from memory

LR images are NEVER written to disk.
"""
import cv2
import os
import json
import time
import logging
import gc
import numpy as np
from typing import List, Dict, Any, Optional

from postflight.dino_embedder import get_embedding, get_mode, release as release_dino
from postflight.lbp_descriptor import get_lbp_histogram
from postflight.hsv_descriptor import get_hsv_histogram
from postflight.fusion_scorer import FusionScorer
from postflight.reference_manager import ReferenceManager, RefEntry

log = logging.getLogger(__name__)

LR_SIZE = (128, 128)


class Matcher:

    def __init__(self, ref_manager: ReferenceManager,
                 matching_cfg: dict,
                 frames_dir: str,
                 matches_dir: str,
                 logs_dir: str,
                 sortie_id: str):
        self._refs      = ref_manager
        self._scorer    = FusionScorer(matching_cfg)
        self._frames_dir = frames_dir
        self._matches_dir = matches_dir
        self._logs_dir   = logs_dir
        self._sortie_id  = sortie_id
        os.makedirs(matches_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        self._raw_detections: List[dict] = []

    def run(self, frame_paths: List[str],
            checkpoint_progress: int = 0) -> List[dict]:
        """
        Match all frames. Returns list of raw detection dicts.
        checkpoint_progress: skip first N frames (resumption).
        """
        total = len(frame_paths)
        log.info(f"Matcher: {total} frames, resuming from {checkpoint_progress}")

        # Check if DINOv2 fell back
        if get_mode() == "mobilenet":
            self._scorer.set_dino_fallback(True)

        feature_types = self._refs.get_feature_types()
        if not feature_types:
            log.error("Matcher: no reference feature types found in refs/")
            return []

        log.info(f"Matcher: feature types = {feature_types}")

        for idx, fpath in enumerate(frame_paths):
            if idx < checkpoint_progress:
                continue

            det = self._process_frame(fpath, idx, total, feature_types)
            if det:
                self._raw_detections.extend(det)

            # Save progress every 10 frames
            if idx % 10 == 0:
                self._save_raw(idx, total)

        self._save_raw(total, total)
        log.info(f"Matcher: complete. {len(self._raw_detections)} raw detections")
        return self._raw_detections

    def _process_frame(self, fpath: str, idx: int, total: int,
                       feature_types: List[str]) -> List[dict]:
        # Load HD frame
        frame_hd = cv2.imread(fpath)
        if frame_hd is None:
            log.warning(f"Cannot read frame: {fpath}")
            return []

        # Parse pose from filename
        pose = self._parse_pose_from_filename(os.path.basename(fpath))
        frame_ts = self._parse_ts_from_filename(os.path.basename(fpath))

        # Downsample to LR in memory — NEVER saved to disk
        frame_lr = cv2.resize(frame_hd, LR_SIZE, interpolation=cv2.INTER_AREA)

        # Extract descriptors once per frame
        try:
            q_dino = get_embedding(frame_lr)
            q_lbp  = get_lbp_histogram(frame_lr)
            q_hsv  = get_hsv_histogram(frame_lr)
        except Exception as e:
            log.error(f"Descriptor extraction failed for {fpath}: {e}")
            del frame_hd, frame_lr
            gc.collect()
            return []

        # Discard LR immediately
        del frame_lr
        gc.collect()

        detections = []

        for ft in feature_types:
            refs = self._refs.get_refs(ft)
            if not refs:
                continue

            # Score against all refs — take maximum (most similar reference wins)
            best_score = 0.0
            best_breakdown = {}

            for ref in refs:
                score, breakdown = self._scorer.score(
                    q_dino, q_lbp, q_hsv,
                    ref.dino, ref.lbp, ref.hsv,
                    feature_type=ft
                )
                if score > best_score:
                    best_score = score
                    best_breakdown = breakdown

            tier = self._scorer.get_tier(best_score)
            if tier == "REJECT":
                continue

            # Save HD proof image
            proof_name = (f"{ft}_{idx:05d}_"
                          f"conf{best_score:.3f}.jpg")
            proof_path = os.path.join(self._matches_dir, proof_name)
            cv2.imwrite(proof_path, frame_hd,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            det = {
                "sortie_id":    self._sortie_id,
                "frame_index":  idx,
                "frame_path":   fpath,
                "feature_type": ft,
                "confidence":   round(best_score, 4),
                "confidence_tier": tier,
                "scores":       best_breakdown,
                "coordinates":  pose,
                "frame_timestamp": frame_ts,
                "proof_image":  proof_path,
            }
            detections.append(det)
            log.info(f"  [{idx+1}/{total}] {ft}: {best_score:.3f} [{tier}] "
                     f"@ ({pose['x']:.2f}, {pose['y']:.2f})")

        # Explicit memory release
        del frame_hd
        gc.collect()

        return detections

    def _save_raw(self, processed: int, total: int):
        out = {
            "sortie_id": self._sortie_id,
            "processed": processed,
            "total": total,
            "detections": self._raw_detections,
        }
        path = os.path.join(self._logs_dir, "raw_detections.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    @staticmethod
    def _parse_pose_from_filename(fname: str) -> dict:
        """Extract x,y,z from filename like frame_1234.567_x3.420_y7.150_z2.800.jpg"""
        import re
        try:
            x = float(re.search(r'x(-?[\d.]+)', fname).group(1))
            y = float(re.search(r'y(-?[\d.]+)', fname).group(1))
            z = float(re.search(r'z(-?[\d.]+)', fname).group(1))
            return {"x": x, "y": y, "z": z}
        except Exception:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

    @staticmethod
    def _parse_ts_from_filename(fname: str) -> float:
        """Extract timestamp from frame_<TS>_..."""
        import re
        try:
            return float(re.search(r'frame_([\d.]+)_', fname).group(1))
        except Exception:
            return 0.0
