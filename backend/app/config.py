"""应用配置加载。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

BACKEND_DIR = Path(__file__).resolve().parent.parent


@dataclass
class InferDefaults:
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    max_det: int = 300


@dataclass
class AppConfig:
    weights_dir: Path
    device: str = "auto"
    max_loaded_models: int = 2
    rescan_interval_sec: float = 30.0
    half: bool = False
    infer: InferDefaults = field(default_factory=InferDefaults)


def load_config(path: Path | None = None) -> AppConfig:
    path = path or (BACKEND_DIR / "config.yaml")
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    infer_raw = raw.get("infer") or {}
    cfg = AppConfig(
        weights_dir=(BACKEND_DIR / str(raw.get("weights_dir", "weights"))).resolve(),
        device=str(raw.get("device", "auto")),
        max_loaded_models=int(raw.get("max_loaded_models", 2)),
        rescan_interval_sec=float(raw.get("rescan_interval_sec", 30)),
        half=bool(raw.get("half", False)),
        infer=InferDefaults(
            conf=float(infer_raw.get("conf", 0.25)),
            iou=float(infer_raw.get("iou", 0.7)),
            imgsz=int(infer_raw.get("imgsz", 640)),
            max_det=int(infer_raw.get("max_det", 300)),
        ),
    )
    return cfg


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
