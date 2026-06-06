"""
postflight/deduplicator.py — Multi-stage duplicate suppression.
Stage 1: Spatial clustering (DBSCAN, radius=0.8m)
Stage 2: Embedding similarity (cosine > 0.95)
Stage 3: Temporal suppression (2-second window)
Output: deduplicated_detections.json + dedup_chain.json
"""
import json
import os
import numpy as np
import logging
from typing import List, Dict, Any
from collections import defaultdict

log = logging.getLogger(__name__)


class Deduplicator:

    def __init__(self, cfg: dict, logs_dir: str, sortie_id: str):
        self._spatial_r  = cfg.get("spatial_cluster_radius_m", 0.8)
        self._emb_thresh = cfg.get("embedding_sim_threshold",  0.95)
        self._temp_sec   = cfg.get("temporal_window_sec",      2.0)
        self._logs_dir   = logs_dir
        self._sortie_id  = sortie_id

    def run(self, raw: List[dict]) -> List[dict]:
        if not raw:
            log.warning("Deduplicator: no raw detections to process")
            return []

        log.info(f"Deduplicator: {len(raw)} raw detections")
        dedup_chain = {}

        # Group by feature type
        by_type: Dict[str, List[dict]] = defaultdict(list)
        for d in raw:
            by_type[d["feature_type"]].append(d)

        canonical = []
        for ft, dets in by_type.items():
            log.info(f"  Processing {ft}: {len(dets)} raw")
            survived = self._stage1_spatial(dets, ft, dedup_chain)
            survived = self._stage2_embedding(survived, ft, dedup_chain)
            survived = self._stage3_temporal(survived, ft, dedup_chain)

            # Assign instance IDs
            for i, d in enumerate(survived):
                d["instance_id"] = f"{ft}_{i+1:03d}"

            canonical.extend(survived)
            log.info(f"  {ft}: {len(dets)} → {len(survived)} canonical")

        # Save outputs
        self._save(canonical, dedup_chain)
        return canonical

    # ── Stage 1: Spatial Clustering ──────────────────────────────────────────

    def _stage1_spatial(self, dets: List[dict], ft: str,
                        chain: dict) -> List[dict]:
        if len(dets) <= 1:
            return dets

        try:
            from sklearn.cluster import DBSCAN
            coords = np.array([
                [d["coordinates"]["x"], d["coordinates"]["y"]] for d in dets
            ])
            labels = DBSCAN(eps=self._spatial_r, min_samples=1).fit_predict(coords)
        except ImportError:
            log.warning("sklearn not installed — spatial dedup using simple distance")
            labels = self._simple_spatial_cluster(dets)

        # Per cluster: keep highest confidence
        clusters: Dict[int, List[dict]] = defaultdict(list)
        for det, lbl in zip(dets, labels):
            clusters[lbl].append(det)

        survived = []
        for lbl, members in clusters.items():
            best = max(members, key=lambda d: d["confidence"])
            suppressed = [m for m in members if m is not best]
            if suppressed:
                chain[best.get("frame_path", "")] = {
                    "stage": "spatial",
                    "absorbed": [s.get("frame_path", "") for s in suppressed],
                }
            survived.append(best)

        return survived

    def _simple_spatial_cluster(self, dets: List[dict]) -> List[int]:
        labels = [-1] * len(dets)
        label = 0
        for i, di in enumerate(dets):
            if labels[i] != -1:
                continue
            labels[i] = label
            xi = di["coordinates"]["x"]; yi = di["coordinates"]["y"]
            for j in range(i+1, len(dets)):
                xj = dets[j]["coordinates"]["x"]; yj = dets[j]["coordinates"]["y"]
                if np.sqrt((xi-xj)**2 + (yi-yj)**2) < self._spatial_r:
                    labels[j] = label
            label += 1
        return labels

    # ── Stage 2: Embedding Similarity ────────────────────────────────────────

    def _stage2_embedding(self, dets: List[dict], ft: str,
                          chain: dict) -> List[dict]:
        if len(dets) <= 1:
            return dets

        # Reload DINO embeddings for survivors
        try:
            from postflight.dino_embedder import get_embedding
            import cv2
        except Exception:
            return dets

        def get_emb(det):
            try:
                img = cv2.imread(det["proof_image"])
                if img is None: return None
                lr = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                return get_embedding(lr)
            except Exception:
                return None

        embeddings = [get_emb(d) for d in dets]
        suppressed_idx = set()

        for i in range(len(dets)):
            if i in suppressed_idx or embeddings[i] is None:
                continue
            for j in range(i+1, len(dets)):
                if j in suppressed_idx or embeddings[j] is None:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                sim = (sim + 1.0) / 2.0  # shift to [0,1]
                if sim > self._emb_thresh:
                    # Suppress lower confidence
                    keep, drop = (i, j) if dets[i]["confidence"] >= dets[j]["confidence"] else (j, i)
                    suppressed_idx.add(drop)
                    chain[dets[keep].get("frame_path","")] = {
                        "stage": "embedding", "sim": round(sim, 3),
                        "absorbed": [dets[drop].get("frame_path","")]
                    }

        return [d for i, d in enumerate(dets) if i not in suppressed_idx]

    # ── Stage 3: Temporal Suppression ────────────────────────────────────────

    def _stage3_temporal(self, dets: List[dict], ft: str,
                         chain: dict) -> List[dict]:
        if len(dets) <= 1:
            return dets

        sorted_dets = sorted(dets, key=lambda d: d.get("frame_timestamp", 0))
        survived = [sorted_dets[0]]

        for det in sorted_dets[1:]:
            ts = det.get("frame_timestamp", 0)
            last_ts = survived[-1].get("frame_timestamp", 0)
            if abs(ts - last_ts) < self._temp_sec:
                # Same temporal window — keep higher confidence
                if det["confidence"] > survived[-1]["confidence"]:
                    chain[det.get("frame_path","")] = {
                        "stage": "temporal",
                        "absorbed": [survived[-1].get("frame_path","")]
                    }
                    survived[-1] = det
            else:
                survived.append(det)

        return survived

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save(self, canonical: List[dict], chain: dict):
        dedup_path = os.path.join(self._logs_dir, "deduplicated_detections.json")
        chain_path = os.path.join(self._logs_dir, "dedup_chain.json")

        with open(dedup_path, "w") as f:
            json.dump({
                "sortie_id":  self._sortie_id,
                "count":      len(canonical),
                "detections": canonical,
            }, f, indent=2)

        with open(chain_path, "w") as f:
            json.dump(chain, f, indent=2)

        log.info(f"Deduplicator: saved {len(canonical)} canonical detections")
