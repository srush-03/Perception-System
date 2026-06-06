"""
revalidation/revalidator.py — Base station (PC) secondary validation.
Uses COMPLETELY DIFFERENT logic from onboard validator:
  - ORB feature matching (not DINOv2)
  - HSV histogram comparison
  - SHA-256 file integrity check
  - Coordinate plausibility check
  - False-positive filter (HSV range deviation)

Run this on the PC after receiving transferred files from Jetson.
"""
import cv2
import os
import json
import hashlib
import logging
import math
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

log = logging.getLogger(__name__)

ORB_MIN_MATCHES   = 15
BHATT_THRESHOLD   = 0.35   # max allowed Bhattacharyya distance
FP_STD_THRESHOLD  = 3.0    # max HSV deviation in std devs


class Revalidator:

    def __init__(self, refs_dir: str = "refs",
                 arena_cfg: dict = None):
        self._refs_dir  = refs_dir
        self._arena_cfg = arena_cfg or {}
        self._orb = cv2.ORB_create(nfeatures=1000)
        self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Preload reference HSV stats for FP filter
        self._ref_hsv_stats = self._compute_ref_hsv_stats()

    def run(self, manifest_path: str) -> dict:
        """
        Full revalidation from transfer manifest.
        Returns final mission verdict dict.
        """
        log.info(f"Revalidator: loading manifest {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        sortie_id = manifest["sortie_id"]
        files     = manifest["files"]
        log.info(f"Revalidator: {len(files)} files to verify")

        results = {
            "sortie_id":    sortie_id,
            "checks":       {},
            "per_detection": [],
        }

        # ── Check 1: File Integrity (SHA-256) ─────────────────────────────
        integrity_failures = []
        for f_entry in files:
            path = f_entry["path"]
            expected_sha = f_entry.get("sha256", "")
            if not os.path.exists(path):
                integrity_failures.append({"file": path, "reason": "missing"})
                continue
            if expected_sha:
                actual = self._sha256(path)
                if actual != expected_sha:
                    integrity_failures.append({"file": path, "reason": "hash_mismatch"})
        results["checks"]["integrity"] = {
            "pass": len(integrity_failures) == 0,
            "failures": integrity_failures,
        }

        # ── Load canonical detections from validation JSON ─────────────────
        val_entry = next((f for f in files if f.get("type") == "validation"), None)
        if not val_entry:
            log.error("No validation JSON in manifest")
            results["verdict"] = "RE_SORTIE_REQUIRED"
            results["reason"]  = "missing_validation_json"
            return results

        with open(val_entry["path"]) as f:
            val_data = json.load(f)
        canonical = val_data.get("checks", {})

        # Load proof images from manifest
        proof_entries = [f for f in files if f.get("feature_type")]

        # ── Check 2: ORB Feature Matching ─────────────────────────────────
        orb_failures = []
        for entry in proof_entries:
            ft    = entry["feature_type"]
            proof = entry["path"]
            img   = cv2.imread(proof)
            if img is None:
                orb_failures.append({"proof": proof, "reason": "unreadable"})
                continue
            matched = self._orb_match_against_refs(img, ft)
            if not matched:
                orb_failures.append({"proof": proof, "feature_type": ft,
                                     "reason": "insufficient_orb_matches"})
        results["checks"]["orb_matching"] = {
            "pass": len(orb_failures) == 0,
            "failures": orb_failures,
        }

        # ── Check 3: HSV Color Histogram ──────────────────────────────────
        hsv_failures = []
        for entry in proof_entries:
            ft    = entry["feature_type"]
            proof = entry["path"]
            img   = cv2.imread(proof)
            if img is None:
                continue
            img_lr = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            matched = self._hsv_match_against_refs(img_lr, ft)
            if not matched:
                hsv_failures.append({"proof": proof, "feature_type": ft})
        results["checks"]["hsv_histogram"] = {
            "pass": len(hsv_failures) == 0,
            "failures": hsv_failures,
        }

        # ── Check 4: Coordinate Plausibility ──────────────────────────────
        x_bounds = self._arena_cfg.get("x_bounds", [-1.0, 11.0])
        y_bounds = self._arena_cfg.get("y_bounds", [-1.0,  8.0])
        coord_failures = []
        # Re-read coordinates from canonical detections JSON
        canon_path = os.path.join(
            os.path.dirname(val_entry["path"]),
            "deduplicated_detections.json"
        )
        if os.path.exists(canon_path):
            with open(canon_path) as f:
                canon_data = json.load(f)
            for det in canon_data.get("detections", []):
                x = det["coordinates"].get("x", 0)
                y = det["coordinates"].get("y", 0)
                if not (x_bounds[0] <= x <= x_bounds[1] and
                        y_bounds[0] <= y <= y_bounds[1]):
                    coord_failures.append(det.get("instance_id", "?"))
        results["checks"]["coordinates"] = {
            "pass": len(coord_failures) == 0,
            "out_of_bounds": coord_failures,
        }

        # ── Check 5: False Positive Filter ────────────────────────────────
        fp_flags = []
        for entry in proof_entries:
            ft    = entry["feature_type"]
            proof = entry["path"]
            img   = cv2.imread(proof)
            if img is None:
                continue
            img_lr = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            if self._is_false_positive(img_lr, ft):
                fp_flags.append({"proof": proof, "feature_type": ft})
        results["checks"]["false_positive_filter"] = {
            "pass": len(fp_flags) == 0,
            "flagged": fp_flags,
        }

        # ── Final Verdict ──────────────────────────────────────────────────
        all_pass = all(c["pass"] for c in results["checks"].values())
        failed_types = set()
        for entry in orb_failures + hsv_failures:
            failed_types.add(entry.get("feature_type", "unknown"))

        results["verdict"] = "MISSION_SUCCESS" if all_pass else "RE_SORTIE_REQUIRED"
        results["target_features"] = sorted(failed_types)

        out_path = os.path.join(
            os.path.dirname(manifest_path),
            f"revalidation_{sortie_id}.json"
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        log.info(f"Revalidation: {results['verdict']}")
        if failed_types:
            log.warning(f"  Failed types: {failed_types}")
        return results

    # ── Internal helpers ──────────────────────────────────────────────────

    def _orb_match_against_refs(self, proof_img: np.ndarray,
                                 feature_type: str) -> bool:
        ref_dir = os.path.join(self._refs_dir, feature_type)
        if not os.path.isdir(ref_dir):
            log.warning(f"No ref dir for {feature_type}")
            return False

        q_gray = cv2.cvtColor(proof_img, cv2.COLOR_BGR2GRAY)
        kp_q, des_q = self._orb.detectAndCompute(q_gray, None)
        if des_q is None:
            return False

        best_matches = 0
        for fname in os.listdir(ref_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ref = cv2.imread(os.path.join(ref_dir, fname))
            if ref is None:
                continue
            r_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            kp_r, des_r = self._orb.detectAndCompute(r_gray, None)
            if des_r is None:
                continue
            matches = self._bf.match(des_q, des_r)
            if len(matches) > best_matches:
                best_matches = len(matches)

        log.debug(f"ORB {feature_type}: best_matches={best_matches}")
        return best_matches >= ORB_MIN_MATCHES

    def _hsv_match_against_refs(self, img_lr: np.ndarray,
                                 feature_type: str) -> bool:
        ref_dir = os.path.join(self._refs_dir, feature_type)
        if not os.path.isdir(ref_dir):
            return False

        q_hsv = self._get_hsv_hist(img_lr)
        best_dist = 1.0

        for fname in os.listdir(ref_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ref = cv2.imread(os.path.join(ref_dir, fname))
            if ref is None:
                continue
            ref_lr  = cv2.resize(ref, (128, 128), interpolation=cv2.INTER_AREA)
            ref_hsv = self._get_hsv_hist(ref_lr)
            dist    = float(cv2.compareHist(q_hsv, ref_hsv,
                                            cv2.HISTCMP_BHATTACHARYYA))
            if dist < best_dist:
                best_dist = dist

        log.debug(f"HSV {feature_type}: best_bhatt={best_dist:.3f}")
        return best_dist < BHATT_THRESHOLD

    def _is_false_positive(self, img_lr: np.ndarray,
                            feature_type: str) -> bool:
        """True if detected image HSV is > FP_STD_THRESHOLD std devs from ref mean."""
        stats = self._ref_hsv_stats.get(feature_type)
        if not stats:
            return False
        hsv = cv2.cvtColor(img_lr, cv2.COLOR_BGR2HSV)
        query_means = np.array([hsv[:,:,c].mean() for c in range(3)])
        z = np.abs((query_means - stats["mean"]) / (stats["std"] + 1e-6))
        return bool(z.max() > FP_STD_THRESHOLD)

    def _compute_ref_hsv_stats(self) -> Dict[str, dict]:
        stats = {}
        if not os.path.isdir(self._refs_dir):
            return stats
        for ft in os.listdir(self._refs_dir):
            ref_dir = os.path.join(self._refs_dir, ft)
            if not os.path.isdir(ref_dir):
                continue
            means = []
            for fname in os.listdir(ref_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(ref_dir, fname))
                if img is None:
                    continue
                lr  = cv2.resize(img, (128, 128))
                hsv = cv2.cvtColor(lr, cv2.COLOR_BGR2HSV)
                means.append([hsv[:,:,c].mean() for c in range(3)])
            if means:
                arr = np.array(means)
                stats[ft] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}
        return stats

    @staticmethod
    def _get_hsv_hist(img_lr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_lr, cv2.COLOR_BGR2HSV)
        h   = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        cv2.normalize(h, h)
        return h

    @staticmethod
    def _sha256(path: str) -> str:
        ha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                ha.update(chunk)
        return ha.hexdigest()
