"""
config_loader.py — Singleton YAML config loader.
All modules import: from config_loader import CFG
"""
import yaml
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config", "system_config.yaml")
_cfg = None


class _Config:
    def __init__(self, data: dict):
        self._data = data

    def get(self, *keys, default=None):
        d = self._data
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, {})
        return d if d != {} else default

    def __getitem__(self, key):
        return self._data[key]

    @property
    def camera(self):        return self._data.get("camera", {})
    @property
    def navigation(self):    return self._data.get("navigation", {})
    @property
    def trigger(self):       return self._data.get("trigger", {})
    @property
    def keyframe(self):      return self._data.get("keyframe", {})
    @property
    def quality(self):       return self._data.get("quality", {})
    @property
    def matching(self):      return self._data.get("matching", {})
    @property
    def deduplication(self): return self._data.get("deduplication", {})
    @property
    def storage(self):       return self._data.get("storage", {})
    @property
    def arena(self):         return self._data.get("arena", {})
    @property
    def boundary(self):      return self._data.get("boundary_detector", {})
    @property
    def paths(self):         return self._data.get("paths", {})
    @property
    def logging(self):       return self._data.get("logging", {})


def load_config(path: str = None) -> _Config:
    global _cfg
    p = path or _CONFIG_PATH
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    _cfg = _Config(data)
    return _cfg


def get_config() -> _Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


CFG = get_config()
