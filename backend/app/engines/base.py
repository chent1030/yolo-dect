"""引擎抽象基类。所有引擎把不同 YOLO 版本的输出归一化为统一 detection dict：

{
    "box": [x1, y1, x2, y2],       # 原图像素坐标
    "score": float,
    "class_id": int,
    "class_name": str,
    "mask": [[x, y], ...] | None,  # 像素坐标多边形，segment 任务
    "keypoints": [[x, y, conf], ...] | None,  # pose 任务
}
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEngine(ABC):
    task: str = "detect"
    names: list[str] = []

    @abstractmethod
    def predict(
        self,
        img_bgr: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
    ) -> list[dict]:
        """对 BGR numpy 图像推理，返回统一格式的 detection 列表。"""

    def unload(self) -> None:
        """释放模型资源（默认无操作）。"""
