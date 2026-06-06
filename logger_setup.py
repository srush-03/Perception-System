"""
logger_setup.py — Centralized logging. Call setup_logging() once in main.py.
"""
import logging
import os
from datetime import datetime


def setup_logging(log_dir: str = "data/logs", level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ascend_{ts}.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)-28s %(message)s",
        datefmt="%H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    root.info(f"Logging started → {log_file}")
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
