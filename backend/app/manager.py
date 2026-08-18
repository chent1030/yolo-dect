"""模型管理器：懒加载 + LRU 淘汰 + 同模型串行锁 + 权重热更新。"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

import numpy as np

from .config import AppConfig, resolve_device
from .engines import BaseEngine, create_engine
from .registry import ModelMeta, ModelRegistry, SceneMeta

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.registry = ModelRegistry(cfg.weights_dir)
        self.device = resolve_device(cfg.device)
        self._scenes: dict[str, SceneMeta] = {}
        self._loaded: OrderedDict[str, tuple[BaseEngine, float]] = OrderedDict()
        self._model_locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # ---- 注册表 ----

    def rescan(self) -> None:
        self._scenes = self.registry.rescan()

    @property
    def scenes(self) -> dict[str, SceneMeta]:
        return self._scenes

    def find_model(self, model_id: str) -> ModelMeta | None:
        scene_name, _, filename = model_id.partition("/")
        scene = self._scenes.get(scene_name)
        if scene is None:
            return None
        return next((m for m in scene.models if m.id == model_id or m.filename == filename), None)

    # ---- 引擎生命周期 ----

    def _lock_for(self, model_id: str) -> threading.Lock:
        with self._global_lock:
            return self._model_locks.setdefault(model_id, threading.Lock())

    def _get_engine(self, meta: ModelMeta) -> BaseEngine:
        with self._global_lock:
            entry = self._loaded.get(meta.id)
            if entry is not None:
                engine, mtime = entry
                if mtime == meta.mtime:  # 权重未变化
                    self._loaded.move_to_end(meta.id)
                    return engine
                engine.unload()
                del self._loaded[meta.id]

        with self._lock_for(meta.id):
            with self._global_lock:
                entry = self._loaded.get(meta.id)
                if entry is not None and entry[1] == meta.mtime:
                    self._loaded.move_to_end(meta.id)
                    return entry[0]

            t0 = time.perf_counter()
            engine = create_engine(meta.engine, meta.path, meta.task, self.device, self.cfg.half)
            logger.info(
                "加载模型 %s 耗时 %.1fs (device=%s)", meta.id, time.perf_counter() - t0, self.device
            )
            with self._global_lock:
                self._loaded[meta.id] = (engine, meta.mtime)
                while len(self._loaded) > self.cfg.max_loaded_models:
                    _, (evicted, _) = self._loaded.popitem(last=False)
                    evicted.unload()
                    logger.info("LRU 淘汰模型释放显存/内存")
            return engine

    def loaded_model_ids(self) -> list[str]:
        with self._global_lock:
            return list(self._loaded.keys())

    # ---- 推理 ----

    def detect(
        self,
        model_id: str,
        img_bgr: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
    ) -> dict:
        meta = self.find_model(model_id)
        if meta is None:
            raise KeyError(f"模型不存在: {model_id}（可能尚未扫描，请先刷新场景列表）")

        engine = self._get_engine(meta)
        h, w = img_bgr.shape[:2]
        with self._lock_for(meta.id):  # 同一模型推理串行化，避免设备争用
            t0 = time.perf_counter()
            detections = engine.predict(img_bgr, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)
            inference_ms = (time.perf_counter() - t0) * 1000

        return {
            "model_id": meta.id,
            "model_name": meta.filename,
            "task": meta.task,
            "device": self.device,
            "image": {"width": w, "height": h},
            "detections": detections,
            "inference_ms": round(inference_ms, 1),
        }
