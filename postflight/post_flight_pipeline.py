"""
postflight/post_flight_pipeline.py — Orchestrates all post-flight stages.
Triggered by: DOCKED_FOR_CHARGING state.
Stages: Stitch → Match → Deduplicate → Validate → Package

Fully checkpoint-resumable: if power is lost mid-stage, restarts from
the last completed checkpoint on next run.
"""
import os
import json
import glob
import logging
import time
from typing import List, Optional

from mission_state import PipelineCheckpoint
from postflight.stitcher     import Stitcher
from postflight.matcher      import Matcher
from postflight.deduplicator import Deduplicator
from postflight.validator    import Validator
from postflight.reference_manager import ReferenceManager

log = logging.getLogger(__name__)


class PostFlightPipeline:

    def __init__(self, cfg: dict, sortie_id: str, ref_manager: ReferenceManager):
        self._cfg        = cfg
        self._sortie_id  = sortie_id
        self._refs       = ref_manager
        self._checkpoint = PipelineCheckpoint(sortie_id)

        self._frames_dir  = cfg["paths"]["frames_dir"]
        self._mosaic_dir  = cfg["paths"]["mosaic_dir"]
        self._matches_dir = cfg["paths"]["matches_dir"]
        self._logs_dir    = cfg["paths"]["logs_dir"]

        for d in [self._frames_dir, self._mosaic_dir,
                  self._matches_dir, self._logs_dir]:
            os.makedirs(d, exist_ok=True)

    def run(self) -> dict:
        """
        Run full post-flight pipeline. Returns final validation result.
        Resumes from checkpoint if partially complete.
        """
        log.info("="*60)
        log.info(f"POST-FLIGHT PIPELINE START: {self._sortie_id}")
        log.info("="*60)
        t_start = time.time()

        # ── Stage 0: Collect frame paths ──────────────────────────────────
        frame_paths = self._get_frame_paths()
        if not frame_paths:
            log.error("No frames found in frames directory!")
            return {"result": "FAILURE", "reason": "no_frames"}

        log.info(f"Stage 0: {len(frame_paths)} keyframes found")
        self._checkpoint.mark("frame_selection_complete", True,
                              **{"frame_count": len(frame_paths)})

        # ── Stage 1: Stitching ────────────────────────────────────────────
        mosaic_path = None
        if not self._checkpoint.is_done("stitching_complete"):
            log.info("Stage 1: Stitching...")
            stitcher = Stitcher(mosaic_dir=self._mosaic_dir)
            mosaic_path = stitcher.stitch(frame_paths, self._sortie_id)
            self._checkpoint.mark("stitching_complete", True,
                                  stitching_output=mosaic_path or "")
            log.info(f"Stage 1: complete → {mosaic_path}")
        else:
            mosaic_path = self._checkpoint.get("stitching_output")
            log.info(f"Stage 1: skipped (already done) → {mosaic_path}")

        # ── Stage 2: Matching ─────────────────────────────────────────────
        raw_detections = []
        if not self._checkpoint.is_done("matching_complete"):
            log.info("Stage 2: Matching...")
            progress = self._checkpoint.get("matching_progress", {})
            resume_at = progress.get("processed", 0) if isinstance(progress, dict) else 0

            matcher = Matcher(
                ref_manager=self._refs,
                matching_cfg=self._cfg["matching"],
                frames_dir=self._frames_dir,
                matches_dir=self._matches_dir,
                logs_dir=self._logs_dir,
                sortie_id=self._sortie_id,
            )
            raw_detections = matcher.run(frame_paths, checkpoint_progress=resume_at)
            self._checkpoint.mark("matching_complete", True,
                                  matching_progress={"processed": len(frame_paths),
                                                     "total": len(frame_paths)})
            log.info(f"Stage 2: complete. {len(raw_detections)} raw detections")
        else:
            raw_detections = self._load_raw_detections()
            log.info(f"Stage 2: skipped. Loaded {len(raw_detections)} from disk")

        # ── Stage 3: Deduplication ────────────────────────────────────────
        canonical = []
        if not self._checkpoint.is_done("dedup_complete"):
            log.info("Stage 3: Deduplication...")
            dedup = Deduplicator(
                cfg=self._cfg["deduplication"],
                logs_dir=self._logs_dir,
                sortie_id=self._sortie_id,
            )
            canonical = dedup.run(raw_detections)
            self._checkpoint.mark("dedup_complete", True)
            log.info(f"Stage 3: {len(raw_detections)} → {len(canonical)} canonical")
        else:
            canonical = self._load_canonical_detections()
            log.info(f"Stage 3: skipped. Loaded {len(canonical)} canonical")

        # ── Stage 4: Validation ───────────────────────────────────────────
        validation_result = {}
        if not self._checkpoint.is_done("validation_complete"):
            log.info("Stage 4: Validation...")
            expected_types = self._refs.get_feature_types()
            validator = Validator(
                arena_cfg=self._cfg["arena"],
                matching_cfg=self._cfg["matching"],
                logs_dir=self._logs_dir,
                sortie_id=self._sortie_id,
            )
            validation_result = validator.validate(canonical, expected_types)
            self._checkpoint.mark("validation_complete", True)
            log.info(f"Stage 4: {validation_result.get('result', 'UNKNOWN')}")
        else:
            validation_result = self._load_validation()
            log.info("Stage 4: skipped (already done)")

        # ── Stage 5: Package for transfer ────────────────────────────────
        if not self._checkpoint.is_done("transfer_ready"):
            log.info("Stage 5: Packaging for transfer...")
            self._package_transfer(canonical, validation_result, mosaic_path)
            self._checkpoint.mark("transfer_ready", True)
            log.info("Stage 5: transfer package ready")
        else:
            log.info("Stage 5: skipped (already packaged)")

        elapsed = time.time() - t_start
        log.info(f"POST-FLIGHT PIPELINE COMPLETE in {elapsed:.1f}s")
        log.info("="*60)
        return validation_result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_frame_paths(self) -> List[str]:
        paths = sorted(
            glob.glob(os.path.join(self._frames_dir, "frame_*.jpg"))
        )
        return paths

    def _load_raw_detections(self) -> list:
        path = os.path.join(self._logs_dir, "raw_detections.json")
        try:
            with open(path) as f:
                return json.load(f).get("detections", [])
        except Exception as e:
            log.error(f"Cannot load raw_detections.json: {e}")
            return []

    def _load_canonical_detections(self) -> list:
        path = os.path.join(self._logs_dir, "deduplicated_detections.json")
        try:
            with open(path) as f:
                return json.load(f).get("detections", [])
        except Exception as e:
            log.error(f"Cannot load deduplicated_detections.json: {e}")
            return []

    def _load_validation(self) -> dict:
        path = os.path.join(self._logs_dir, "onboard_validation.json")
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _package_transfer(self, canonical: list, validation: dict,
                           mosaic_path: Optional[str]):
        """Write transfer manifest listing all files to send to base station."""
        import hashlib

        transfer_dir = os.path.join("data", "transfer")
        os.makedirs(transfer_dir, exist_ok=True)

        files = []
        for det in canonical:
            proof = det.get("proof_image", "")
            if os.path.exists(proof):
                sha = self._sha256(proof)
                files.append({"path": proof, "sha256": sha,
                              "feature_type": det["feature_type"],
                              "instance_id": det["instance_id"]})

        # Include validation JSON
        val_path = os.path.join(self._logs_dir, "onboard_validation.json")
        if os.path.exists(val_path):
            files.append({"path": val_path, "sha256": self._sha256(val_path),
                          "type": "validation"})

        manifest = {
            "sortie_id":   self._sortie_id,
            "timestamp":   time.time(),
            "files":       files,
            "detection_count": len(canonical),
            "validation_result": validation.get("result", "UNKNOWN"),
        }
        mpath = os.path.join(transfer_dir, f"manifest_{self._sortie_id}.json")
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"Transfer manifest: {mpath}")

    @staticmethod
    def _sha256(path: str) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
