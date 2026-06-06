"""
postflight/dino_embedder.py — DINOv2 ViT-S/14 feature extractor.
Input: 128x128 BGR numpy array.
Output: 384-dim L2-normalized float32 vector.

Auto-falls back to MobileNetV2 (1280-dim) if DINOv2 fails to load.
Fallback weights are adjusted in fusion_scorer.py automatically.
"""
import numpy as np
import logging
import gc
from alert_writer import alert_dino_fallback

log = logging.getLogger(__name__)

# Lazy-loaded — only imported when first needed
_model  = None
_preprocess = None
_device = None
_mode   = None   # 'dino' | 'mobilenet'


def _load_dino():
    global _model, _preprocess, _device, _mode
    import torch
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"DINOv2: loading ViT-S/14 on {_device}")
    try:
        _model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False
        )
        _model = _model.to(_device).eval()

        from torchvision import transforms
        _preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        _mode = "dino"
        log.info("DINOv2 ViT-S/14 loaded successfully")

    except Exception as e:
        log.warning(f"DINOv2 load failed ({e}) — activating MobileNetV2 fallback")
        alert_dino_fallback(str(e))
        _load_mobilenet()


def _load_mobilenet():
    global _model, _preprocess, _device, _mode
    import torch
    import torchvision.models as models
    from torchvision import transforms

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = models.mobilenet_v2(pretrained=True)
    # Use features only (no classifier)
    _model = torch.nn.Sequential(*list(base.children())[:-1],
                                  torch.nn.AdaptiveAvgPool2d(1))
    _model = _model.to(_device).eval()

    _preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    _mode = "mobilenet"
    log.info("MobileNetV2 fallback loaded")


def get_embedding(bgr_128: np.ndarray) -> np.ndarray:
    """
    bgr_128: (128, 128, 3) uint8 BGR image.
    Returns: 1-D float32 L2-normalized embedding.
    """
    global _model, _preprocess, _device, _mode
    if _model is None:
        _load_dino()

    import torch
    import cv2
    rgb = cv2.cvtColor(bgr_128, cv2.COLOR_BGR2RGB)
    tensor = _preprocess(rgb).unsqueeze(0).to(_device)

    with torch.no_grad():
        if _mode == "dino":
            feat = _model(tensor)          # (1, 384)
        else:
            feat = _model(tensor).squeeze() # (1280,)

    vec = feat.cpu().numpy().flatten().astype(np.float32)
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


def get_mode() -> str:
    """Returns 'dino' or 'mobilenet' — used by fusion_scorer for weight adjustment."""
    return _mode or "unknown"


def release():
    """Free GPU memory after post-flight processing."""
    global _model, _preprocess, _device, _mode
    import torch
    _model = None
    _preprocess = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("DINOv2 embedder: GPU memory released")
