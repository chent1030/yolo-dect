"""引擎适配层：根据 checkpoint 元信息自动识别 YOLO 版本并选择引擎。

- ultralytics 引擎: YOLOv8 / v9 / v10 / v11 / v12（同一 API，det/seg/pose 全支持）
- yolov5 引擎:     YOLOv5 系（依赖 third_party/yolov5 本地仓库，det/seg）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from ..config import BACKEND_DIR
from .base import BaseEngine
from .ultralytics_engine import UltralyticsEngine
from .yolov5_engine import YoloV5Engine

KNOWN_TASKS = {"detect", "segment", "pose"}


def ensure_yolov5_importable() -> bool:
    """v5 checkpoint 的 pickle 引用 yolov5 仓库的 models 包，unpickle 前需把仓库根目录加入 sys.path。"""
    repo = BACKEND_DIR / "third_party" / "yolov5"
    if repo.exists() and str(repo) not in sys.path:
        sys.path.append(str(repo))
    return repo.exists()


def _normalize_names(names) -> list[str]:
    """v5 的 names 是 list，ultralytics 是 dict {id: name}，统一成有序 list。"""
    if isinstance(names, dict):
        ids = sorted(k for k in names if isinstance(k, int))
        size = (ids[-1] + 1) if ids else 0
        return [str(names.get(i, f"class_{i}")) for i in range(size)]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return []


def peek_checkpoint(path: Path) -> dict:
    # 权重来自本地受控目录，checkpoint 内含 pickle 代码，必须 weights_only=False 才能读出元信息。
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"不认识的 checkpoint 格式: {path.name}")
    return ckpt


def _task_from_model(model) -> str | None:
    task = getattr(model, "task", None)
    if isinstance(task, str) and task in KNOWN_TASKS:
        return task
    # v5 仓库的模型类名可判断任务
    cls_name = type(model).__name__
    if "Segmentation" in cls_name:
        return "segment"
    if "Pose" in cls_name:
        return "pose"
    if "Classification" in cls_name:
        return None
    yaml_dict = getattr(model, "yaml", None) or {}
    if isinstance(yaml_dict, dict):
        if yaml_dict.get("task") in KNOWN_TASKS:
            return yaml_dict["task"]
        # ultralytics pose/seg 的 yaml head 里带 kpt_shape / mask_ratio 等特征
        if "kpt_shape" in yaml_dict:
            return "pose"
        head = yaml_dict.get("head") or []
        for layer in head:
            if isinstance(layer, (list, tuple)) and len(layer) >= 3:
                module = str(layer[-1]).lower()
                if "segment" in module or module == "segment":
                    return "segment"
    return None


def sniff_model(path: Path) -> tuple[str, str, list[str]]:
    """返回 (engine, task, class_names)。加载完整引擎前只做一次轻量 peek。"""
    ensure_yolov5_importable()
    try:
        ckpt = peek_checkpoint(path)
    except ModuleNotFoundError as e:
        if not ensure_yolov5_importable():
            raise RuntimeError(
                f"疑似 YOLOv5 权重，但缺少 yolov5 仓库（{BACKEND_DIR / 'third_party' / 'yolov5'}），"
                "请 git clone 后重试"
            ) from e
        raise
    model = ckpt.get("model", None) or ckpt.get("ema", None)

    engine = None
    if "train_args" in ckpt:  # ultralytics 训练产物独有
        engine = "ultralytics"
    yaml_dict = getattr(model, "yaml", None) if model is not None else None
    if engine is None and isinstance(yaml_dict, dict):
        if "anchors" in yaml_dict:  # v5 系 yaml 带 anchors
            engine = "yolov5"
        elif "yaml_file" in yaml_dict or "task" in yaml_dict:
            engine = "ultralytics"
    if engine is None and model is not None:
        module = type(model).__module__ or ""
        engine = "ultralytics" if "ultralytics" in module else "yolov5"

    if engine is None:
        # 仍未识别（如 v5 半精度模型属性缺失），优先交给 ultralytics 尝试
        engine = "ultralytics"

    task = _task_from_model(model) if model is not None else None
    if task is None:
        stem = path.stem.lower()
        if stem.endswith(("-seg", "-segment")):
            task = "segment"
        elif stem.endswith("-pose"):
            task = "pose"
        else:
            task = "detect"

    names = _normalize_names(getattr(model, "names", None)) if model is not None else []
    return engine, task, names


def create_engine(engine: str, path: Path, task: str, device: str, half: bool) -> BaseEngine:
    """按 registry 嗅探出的引擎名创建对应引擎。"""
    if engine == "yolov5":
        return YoloV5Engine(path=path, task=task, device=device, half=half)
    return UltralyticsEngine(path=path, task=task, device=device, half=half)
