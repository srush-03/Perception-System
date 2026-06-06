"""
postflight/reference_manager.py — Loads and caches reference seed embeddings.

Auto-discovers feature types from subdirectories in refs/.
Precomputes DINOv2 + LBP + HSV for every reference image.
Cache invalidated when any reference file changes (mtime check).
Hot-swappable: call reload() to pick up new seeds without restart.
"""
import os
import pickle
import time
import logging
import cv2
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field

from postflight.dino_embedder import get_embedding
from postflight.lbp_descriptor import get_lbp_histogram
from postflight.hsv_descriptor import get_hsv_histogram

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class RefEntry:
    feature_type: str
    filename:     str
    dino:         np.ndarray
    lbp:          np.ndarray
    hsv:          np.ndarray


class ReferenceManager:

    def __init__(self, refs_dir: str = "refs", cache_dir: str = "cache"):
        self._refs_dir  = refs_dir
        self._cache_file = os.path.join(cache_dir, "ref_embeddings.pkl")
        os.makedirs(cache_dir, exist_ok=True)
        self._entries: Dict[str, List[RefEntry]] = {}  # feature_type → [entries]
        self._file_mtimes: Dict[str, float] = {}

    def load(self) -> int:
        """Load or rebuild cache. Returns total number of reference images loaded."""
        if self._is_cache_valid():
            self._load_cache()
        else:
            self._build_cache()
        total = sum(len(v) for v in self._entries.values())
        log.info(f"ReferenceManager: {len(self._entries)} feature types, "
                 f"{total} reference images")
        return total

    def reload(self):
        """Hot-reload: rebuild embeddings for any changed files."""
        self._build_cache()

    def get_feature_types(self) -> List[str]:
        return list(self._entries.keys())

    def get_refs(self, feature_type: str) -> List[RefEntry]:
        return self._entries.get(feature_type, [])

    # ── Cache management ─────────────────────────────────────────────────────

    def _discover_files(self) -> Dict[str, List[str]]:
        """Scan refs/ and return {feature_type: [abs_paths]}."""
        found = {}
        if not os.path.isdir(self._refs_dir):
            log.warning(f"refs dir not found: {self._refs_dir}")
            return found
        for entry in sorted(os.scandir(self._refs_dir)):
            if not entry.is_dir():
                continue
            ft = entry.name
            imgs = []
            for f in sorted(os.scandir(entry.path)):
                if f.is_file() and os.path.splitext(f.name)[1].lower() in SUPPORTED_EXTS:
                    imgs.append(f.path)
            if imgs:
                found[ft] = imgs
                log.info(f"  Found {len(imgs)} refs for '{ft}'")
            else:
                log.warning(f"  No images in refs/{ft}/ — skipping")
        return found

    def _is_cache_valid(self) -> bool:
        if not os.path.exists(self._cache_file):
            return False
        try:
            with open(self._cache_file, "rb") as f:
                saved = pickle.load(f)
            saved_mtimes = saved.get("mtimes", {})
            # Scan current mtimes
            files = self._discover_files()
            for ft, paths in files.items():
                for p in paths:
                    current_mt = os.path.getmtime(p)
                    if saved_mtimes.get(p) != current_mt:
                        log.info(f"Cache stale: {p} changed")
                        return False
            return True
        except Exception:
            return False

    def _load_cache(self):
        with open(self._cache_file, "rb") as f:
            saved = pickle.load(f)
        self._entries     = saved["entries"]
        self._file_mtimes = saved["mtimes"]
        log.info(f"ReferenceManager: loaded cache from {self._cache_file}")

    def _build_cache(self):
        log.info("ReferenceManager: building embedding cache...")
        files = self._discover_files()
        entries = {}
        mtimes  = {}
        t0 = time.time()

        for ft, paths in files.items():
            entries[ft] = []
            for path in paths:
                img = cv2.imread(path)
                if img is None:
                    log.warning(f"Cannot read reference image: {path} — skipping")
                    continue
                # Resize to 128x128 for descriptors
                img_lr = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                try:
                    dino_vec = get_embedding(img_lr)
                    lbp_vec  = get_lbp_histogram(img_lr)
                    hsv_vec  = get_hsv_histogram(img_lr)
                    entries[ft].append(RefEntry(
                        feature_type=ft,
                        filename=os.path.basename(path),
                        dino=dino_vec, lbp=lbp_vec, hsv=hsv_vec,
                    ))
                    mtimes[path] = os.path.getmtime(path)
                    log.info(f"  Embedded: {ft}/{os.path.basename(path)}")
                except Exception as e:
                    log.error(f"  Embedding failed for {path}: {e}")

        self._entries     = entries
        self._file_mtimes = mtimes

        with open(self._cache_file, "wb") as f:
            pickle.dump({"entries": entries, "mtimes": mtimes}, f)

        elapsed = time.time() - t0
        log.info(f"Reference cache built in {elapsed:.1f}s → {self._cache_file}")
