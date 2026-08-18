"""API 数据模型（统一三种任务的结果 schema）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    width: int
    height: int


class Detection(BaseModel):
    box: list[float] = Field(description="[x1, y1, x2, y2] 原图像素坐标")
    score: float
    class_id: int
    class_name: str
    mask: list[list[float]] | None = Field(
        default=None, description="分割多边形顶点 [[x, y], ...]，仅 segment 任务"
    )
    keypoints: list[list[float]] | None = Field(
        default=None, description="关键点 [[x, y, conf], ...]，仅 pose 任务"
    )


class DetectResponse(BaseModel):
    model_id: str
    model_name: str
    task: str
    device: str
    image: ImageInfo
    detections: list[Detection]
    inference_ms: float


class ModelInfo(BaseModel):
    id: str
    scene: str
    filename: str
    task: str
    engine: str
    classes: list[str]
    size_mb: float
    description: str = ""
    defaults: dict = Field(default_factory=dict)


class SceneInfo(BaseModel):
    name: str
    description: str = ""
    models: list[ModelInfo]
