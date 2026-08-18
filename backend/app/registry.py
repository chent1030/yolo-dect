"""模型注册表：扫描 weights/<场景>/*.pt，嗅探引擎/任务/类别。

场景内可选放 scene.yaml 覆盖元信息：
    description: 场景描述
    models:
      helmet_v8s.pt:
        description: 模型描述
        conf: 0.35        # 该模型默认推理参数
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .engines import sniff_model

logger = logging.getLogger(__name__)


@dataclass
class ModelMeta:
    id: str  # "<scene>/<filename>"
    scene: str
    filename: str
    path: Path
    engine: str  # ultralytics | yolov5
    task: str  # detect | segment | pose
    classes: list[str]
    size_mb: float
    mtime: float
    description: str = ""
    defaults: dict = field(default_factory=dict)


@dataclass
class SceneMeta:
    name: str
    description: str = ""
    models: list[ModelMeta] = field(default_factory=list)


class ModelRegistry:
    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self._cache: dict[str, tuple[float, ModelMeta]] = {}  # path -> (mtime, meta)
        self._lock = threading.Lock()

    def rescan(self) -> dict[str, SceneMeta]:
        scenes: dict[str, SceneMeta] = {}
        if self.weights_dir.exists():
            for scene_dir in sorted(p for p in self.weights_dir.iterdir() if p.is_dir()):
                scene = self._scan_scene(scene_dir)
                if scene.models:
                    scenes[scene.name] = scene
        with self._lock:
            live_paths = {m.path for s in scenes.values() for m in s.models}
            self._cache = {
                k: v for k, v in self._cache.items() if Path(k) in live_paths
            }
        return scenes

    def _scan_scene(self, scene_dir: Path) -> SceneMeta:
        cfg = self._load_scene_yaml(scene_dir)
        scene = SceneMeta(name=scene_dir.name, description=str(cfg.get("description", "")))
        model_cfgs = cfg.get("models") or {}
        for pt in sorted(scene_dir.glob("*.pt")):
            override = model_cfgs.get(pt.name) or {}
            meta = self._build_meta(pt, scene_dir.name, override)
            if meta is not None:
                scene.models.append(meta)
        return scene

    def _build_meta(self, pt: Path, scene: str, override: dict) -> ModelMeta | None:
        mtime = pt.stat().st_mtime
        cached = self._cache.get(str(pt))
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            engine, task, classes = sniff_model(pt)
        except Exception as e:
            logger.warning("跳过无法解析的权重 %s: %s", pt, e)
            return None

        defaults = {k: v for k, v in override.items() if k in ("conf", "iou", "imgsz", "max_det")}
        meta = ModelMeta(
            id=f"{scene}/{pt.name}",
            scene=scene,
            filename=pt.name,
            path=pt,
            engine=engine,
            task=task,
            classes=classes,
            size_mb=round(pt.stat().st_size / 1024 / 1024, 1),
            mtime=mtime,
            description=str(override.get("description", "")),
            defaults=defaults,
        )
        with self._lock:
            self._cache[str(pt)] = (mtime, meta)
        logger.info("注册模型 %s [%s/%s, %d 类]", meta.id, engine, task, len(classes))
        return meta

    @staticmethod
    def _load_scene_yaml(scene_dir: Path) -> dict:
        yaml_path = scene_dir / "scene.yaml"
        if not yaml_path.exists():
            return {}
        try:
            return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("解析 %s 失败: %s", yaml_path, e)
            return {}
