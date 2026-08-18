"""API 路由。"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from .schemas import DetectResponse, SceneInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

VALID_IMGSZ = {320, 416, 512, 640, 800, 960, 1280}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _decode_image_bgr(data: bytes) -> np.ndarray:
    import io

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图片文件: {e}") from e
    arr = np.asarray(img)
    return np.ascontiguousarray(arr[:, :, ::-1])  # RGB -> BGR


@router.get("/scenes")
def list_scenes(request: Request) -> dict:
    manager = request.app.state.manager
    scenes = [
        SceneInfo(
            name=s.name,
            description=s.description,
            models=[
                {
                    "id": m.id,
                    "scene": m.scene,
                    "filename": m.filename,
                    "task": m.task,
                    "engine": m.engine,
                    "classes": m.classes,
                    "size_mb": m.size_mb,
                    "description": m.description,
                    "defaults": m.defaults,
                }
                for m in s.models
            ],
        )
        for s in sorted(manager.scenes.values(), key=lambda x: x.name)
    ]
    return {"scenes": [s.model_dump() for s in scenes]}


@router.post("/detect", response_model=DetectResponse)
def detect(
    request: Request,
    file: UploadFile = File(..., description="待检测图片"),
    model_id: str = Form(..., description="场景/模型文件名，如 demo/yolov8n.pt"),
    conf: float | None = Form(None, description="置信度阈值"),
    iou: float | None = Form(None, description="NMS IoU 阈值"),
    imgsz: int | None = Form(None, description="推理尺寸"),
    max_det: int | None = Form(None, description="最大检测数"),
) -> DetectResponse:
    manager = request.app.state.manager
    meta = manager.find_model(model_id)
    model_defaults = meta.defaults if meta is not None else {}
    defaults = manager.cfg.infer

    # 优先级: 请求参数 > 模型默认(scene.yaml) > 全局默认(config.yaml)
    conf = conf if conf is not None else model_defaults.get("conf", defaults.conf)
    iou = iou if iou is not None else model_defaults.get("iou", defaults.iou)
    imgsz = imgsz if imgsz is not None else model_defaults.get("imgsz", defaults.imgsz)
    max_det = max_det if max_det is not None else model_defaults.get("max_det", defaults.max_det)

    conf = min(max(conf, 0.0), 1.0)
    iou = min(max(iou, 0.1), 1.0)
    imgsz = min(VALID_IMGSZ, key=lambda s: abs(s - imgsz))
    max_det = int(min(max(max_det, 1), 1000))

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传的图片为空")
    img_bgr = _decode_image_bgr(data)

    try:
        result = manager.detect(model_id, img_bgr, conf, iou, imgsz, max_det)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("推理失败: %s", model_id)
        raise HTTPException(status_code=500, detail=f"推理失败: {e}") from e

    result["conf"] = conf
    result["iou"] = iou
    result["imgsz"] = imgsz
    return DetectResponse(**result)


@router.post("/admin/rescan")
def rescan(request: Request) -> dict:
    manager = request.app.state.manager
    manager.rescan()
    total = sum(len(s.models) for s in manager.scenes.values())
    return {"scenes": len(manager.scenes), "models": total}


@router.get("/sample-image")
def sample_image(request: Request):
    """返回场景目录里的一张示例图片，方便快速试用。"""
    manager = request.app.state.manager
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if manager.cfg.weights_dir.exists():
        for scene_dir in sorted(manager.cfg.weights_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for f in sorted(scene_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    return FileResponse(f, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="场景目录中没有示例图片")


@router.get("/sample-video")
def sample_video(request: Request):
    """返回场景目录里的一段示例视频。"""
    manager = request.app.state.manager
    if manager.cfg.weights_dir.exists():
        for scene_dir in sorted(manager.cfg.weights_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for f in sorted(scene_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    return FileResponse(f, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="场景目录中没有示例视频")


# ---- 视频整段分析（导出标注视频） ----


@router.post("/video/analyze")
def video_analyze(
    request: Request,
    file: UploadFile = File(..., description="待分析视频"),
    model_id: str = Form(...),
    conf: float | None = Form(None),
    iou: float | None = Form(None),
    imgsz: int | None = Form(None),
    max_det: int | None = Form(None),
    frame_step: int = Form(1, description="每 N 帧检测一次，中间帧复用上次结果"),
) -> dict:
    manager = request.app.state.manager
    tasks = request.app.state.video_tasks
    meta = manager.find_model(model_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    model_defaults = meta.defaults
    defaults = manager.cfg.infer
    conf = conf if conf is not None else model_defaults.get("conf", defaults.conf)
    iou = iou if iou is not None else model_defaults.get("iou", defaults.iou)
    imgsz = imgsz if imgsz is not None else model_defaults.get("imgsz", defaults.imgsz)
    max_det = max_det if max_det is not None else model_defaults.get("max_det", defaults.max_det)
    params = {
        "conf": min(max(conf, 0.0), 1.0),
        "iou": min(max(iou, 0.1), 1.0),
        "imgsz": min(VALID_IMGSZ, key=lambda s: abs(s - imgsz)),
        "max_det": int(min(max(max_det, 1), 1000)),
        "frame_step": int(min(max(frame_step, 1), 30)),
    }

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传的视频为空")
    suffix = "." + (file.filename or "v.mp4").rsplit(".", 1)[-1].lower()
    if suffix not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的视频格式: {suffix}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="yolo_upload_")
    tmp.write(data)
    tmp.close()

    try:
        task = tasks.create(
            video_path=Path(tmp.name),
            source_name=file.filename or "video.mp4",
            model_id=model_id,
            params=params,
        )
    except Exception as e:
        os.unlink(tmp.name)
        raise HTTPException(status_code=400, detail=str(e)) from e
    return tasks.snapshot(task)


@router.get("/video/tasks/{task_id}")
def video_task_status(request: Request, task_id: str) -> dict:
    task = request.app.state.video_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return request.app.state.video_tasks.snapshot(task)


@router.post("/video/tasks/{task_id}/cancel")
def video_task_cancel(request: Request, task_id: str) -> dict:
    tasks = request.app.state.video_tasks
    if not tasks.cancel(task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"ok": True}


@router.get("/video/tasks/{task_id}/download")
def video_task_download(request: Request, task_id: str):
    task = request.app.state.video_tasks.get(task_id)
    if task is None or task.status != "done" or task.output_path is None or not task.output_path.exists():
        raise HTTPException(status_code=404, detail="结果尚未就绪或任务不存在")
    name = "annotated_" + (task.source_name.rsplit(".", 1)[0] or "video") + ".mp4"
    return FileResponse(task.output_path, media_type="video/mp4", filename=name)


@router.get("/health")
def health(request: Request) -> dict:
    manager = request.app.state.manager
    return {
        "status": "ok",
        "device": manager.device,
        "scenes": len(manager.scenes),
        "models": sum(len(s.models) for s in manager.scenes.values()),
        "loaded_models": manager.loaded_model_ids(),
    }
