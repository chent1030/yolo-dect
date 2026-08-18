"""Ultralytics 引擎：YOLOv8 / v9 / v10 / v11 / v12，det/seg/pose 同一 API。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseEngine


class UltralyticsEngine(BaseEngine):
    def __init__(self, path: Path, task: str, device: str, half: bool = False):
        from ultralytics import YOLO

        self._yolo = YOLO(str(path), task=task)
        self.task = self._yolo.task or task
        self.names = self._normalize_names(self._yolo.names)
        self.device = device
        self._yolo.to(device)
        if half and device.startswith("cuda"):
            self._yolo.model.half()

    @staticmethod
    def _normalize_names(names) -> list[str]:
        if isinstance(names, dict):
            ids = sorted(k for k in names if isinstance(k, int))
            size = (ids[-1] + 1) if ids else 0
            return [str(names.get(i, f"class_{i}")) for i in range(size)]
        return [str(n) for n in (names or [])]

    def predict(
        self, img_bgr: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int
    ) -> list[dict]:
        result = self._yolo.predict(
            source=img_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
        )[0]

        h, w = img_bgr.shape[:2]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        masks_xyn = None
        if result.masks is not None and len(result.masks) > 0:
            masks_xyn = result.masks.xyn  # 每个实例一个归一化多边形

        kpts_xy = kpts_conf = None
        if result.keypoints is not None and len(result.keypoints) > 0:
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )

        detections: list[dict] = []
        for i in range(len(boxes)):
            det = {
                "box": [float(v) for v in xyxy[i]],
                "score": float(scores[i]),
                "class_id": int(classes[i]),
                "class_name": self.names[classes[i]] if classes[i] < len(self.names) else f"class_{classes[i]}",
                "mask": None,
                "keypoints": None,
            }
            if masks_xyn is not None and i < len(masks_xyn):
                poly = masks_xyn[i]
                if poly is not None and len(poly) > 0:
                    det["mask"] = [
                        [float(x) * w, float(y) * h] for x, y in np.asarray(poly)
                    ]
            if kpts_xy is not None:
                conf_row = kpts_conf[i] if kpts_conf is not None else None
                det["keypoints"] = [
                    [
                        float(kpts_xy[i][j][0]),
                        float(kpts_xy[i][j][1]),
                        float(conf_row[j]) if conf_row is not None else 1.0,
                    ]
                    for j in range(len(kpts_xy[i]))
                ]
            detections.append(det)
        return detections

    def unload(self) -> None:
        import torch

        self._yolo.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
