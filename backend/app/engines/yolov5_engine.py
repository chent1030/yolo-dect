"""YOLOv5 引擎：通过 third_party/yolov5 本地仓库的 torch.hub 'custom' 加载。

支持 detect / segment；v5-pose 未在本项目验证，显式报不支持。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import BACKEND_DIR
from .base import BaseEngine

logger = logging.getLogger(__name__)

YOLOV5_REPO = BACKEND_DIR / "third_party" / "yolov5"


def _normalize_names(names) -> list[str]:
    """v5 hub 模型的 names 是 dict {id: name}，统一成有序 list。"""
    if isinstance(names, dict):
        ids = sorted(k for k in names if isinstance(k, int))
        size = (ids[-1] + 1) if ids else 0
        return [str(names.get(i, f"class_{i}")) for i in range(size)]
    return [str(n) for n in (names or [])]


class YoloV5Engine(BaseEngine):
    def __init__(self, path: Path, task: str, device: str, half: bool = False):
        import torch

        if not YOLOV5_REPO.exists():
            raise RuntimeError(
                f"YOLOv5 权重需要本地 yolov5 仓库支持，未找到 {YOLOV5_REPO}。"
                "请执行: git clone --depth 1 https://github.com/ultralytics/yolov5 "
                f"{YOLOV5_REPO}"
            )

        self.task = task
        if task == "pose":
            raise NotImplementedError("YOLOv5 pose 模型暂不支持，建议导出为 ultralytics 格式或使用 v8+ 权重")

        # source='local' 跳过 GitHub 下载；skip_validation 跳过其 requirements.txt 检查（与本项目环境冲突）
        self._model = torch.hub.load(
            str(YOLOV5_REPO),
            "custom",
            path=str(path),
            source="local",
            device=device,
            skip_validation=True,
            _verbose=False,
        )
        names = getattr(self._model, "names", None)
        self.names = _normalize_names(names)
        if half and device.startswith("cuda"):
            self._model.half()
        logger.info("YOLOv5 模型已加载: %s (task=%s, device=%s)", path.name, task, device)

    def predict(
        self, img_bgr: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int
    ) -> list[dict]:
        h, w = img_bgr.shape[:2]
        # v5 AutoShape 的阈值参数是模型属性而非 forward 入参
        self._model.conf = conf
        self._model.iou = iou
        self._model.max_det = max_det
        result = self._model(img_bgr, size=imgsz)

        pred = result.xyxy[0].cpu().numpy()  # (N, 6): x1,y1,x2,y2,score,cls（原图坐标）
        masks_xyn = None
        if self.task == "segment":
            masks = getattr(result, "masks", None)
            if masks is not None and len(masks) > 0:
                masks_xyn = masks.xyn

        detections: list[dict] = []
        for i, row in enumerate(pred):
            class_id = int(row[5])
            det = {
                "box": [float(v) for v in row[:4]],
                "score": float(row[4]),
                "class_id": class_id,
                "class_name": self.names[class_id] if class_id < len(self.names) else f"class_{class_id}",
                "mask": None,
                "keypoints": None,
            }
            if masks_xyn is not None and i < len(masks_xyn):
                poly = masks_xyn[i]
                if poly is not None and len(poly) > 0:
                    det["mask"] = [
                        [float(x) * w, float(y) * h] for x, y in np.asarray(poly)
                    ]
            detections.append(det)
        return detections

    def unload(self) -> None:
        import torch

        self._model.to("cpu")
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
