"""视频分析任务管理：上传视频 → 后台线程逐帧检测渲染 → 导出标注 mp4。

任务为内存态（内部工具够用），结果文件落在 outputs/ 目录。
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .config import BACKEND_DIR
from .manager import ModelManager
from .video_annotator import draw_detections

logger = logging.getLogger(__name__)

OUTPUT_DIR = BACKEND_DIR / "outputs"
MAX_FRAMES = 3000  # 安全上限，防止误传超长视频占死 GPU


@dataclass
class VideoTask:
    id: str
    model_id: str
    source_name: str
    video_path: Path
    params: dict
    status: str = "pending"  # pending | running | done | error | cancelled
    total_frames: int = 0
    processed: int = 0
    detections_count: int = 0
    by_class: dict = field(default_factory=dict)
    error: str = ""
    output_path: Path | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_event = threading.Event()


class VideoTaskManager:
    def __init__(self, model_manager: ModelManager):
        self.models = model_manager
        self._tasks: dict[str, VideoTask] = {}
        self._lock = threading.Lock()

    # ---- 任务生命周期 ----

    def create(self, video_path: Path, source_name: str, model_id: str, params: dict) -> VideoTask:
        # 先验证模型，避免起线程后才失败
        if self.models.find_model(model_id) is None:
            raise KeyError(f"模型不存在: {model_id}")
        task = VideoTask(
            id=uuid.uuid4().hex[:12],
            model_id=model_id,
            source_name=source_name,
            video_path=video_path,
            params=params,
        )
        with self._lock:
            self._tasks[task.id] = task
        threading.Thread(target=self._run, args=(task,), daemon=True, name=f"video-{task.id}").start()
        return task

    def get(self, task_id: str) -> VideoTask | None:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.cancel_event.set()
        return True

    def snapshot(self, task: VideoTask) -> dict:
        return {
            "id": task.id,
            "status": task.status,
            "model_id": task.model_id,
            "source_name": task.source_name,
            "total_frames": task.total_frames,
            "processed": task.processed,
            "percent": round(task.processed / task.total_frames * 100, 1) if task.total_frames else 0,
            "detections_count": task.detections_count,
            "by_class": task.by_class,
            "error": task.error,
            "download_url": f"/api/video/tasks/{task.id}/download" if task.status == "done" else None,
        }

    # ---- 分析线程 ----

    def _run(self, task: VideoTask) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        task.status = "running"
        try:
            self._analyze(task)
            task.status = "done"
        except Exception as e:  # noqa: BLE001 后台线程需兜底记录
            logger.exception("视频分析任务 %s 失败", task.id)
            task.status = "error"
            task.error = str(e)
            if task.output_path is not None:
                task.output_path.unlink(missing_ok=True)
        finally:
            task.finished_at = time.time()

    def _analyze(self, task: VideoTask) -> None:
        cap = cv2.VideoCapture(str(task.video_path))
        if not cap.isOpened():
            raise ValueError("无法读取视频文件（格式不受 OpenCV 支持）")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        task.total_frames = total

        raw_output = OUTPUT_DIR / f"{task.id}_raw.mp4"
        writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("VideoWriter 初始化失败")

        frame_step = max(1, int(task.params.get("frame_step", 1)))
        last_detections: list[dict] = []
        try:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or idx >= MAX_FRAMES:
                    break
                if task.cancel_event.is_set():
                    task.status = "cancelled"
                    break
                if idx % frame_step == 0:
                    detections = self.models.detect(
                        task.model_id, frame,
                        conf=task.params["conf"], iou=task.params["iou"],
                        imgsz=task.params["imgsz"], max_det=task.params["max_det"],
                    )["detections"]
                    last_detections = detections
                    task.detections_count += len(detections)
                    for det in detections:
                        task.by_class[det["class_name"]] = task.by_class.get(det["class_name"], 0) + 1
                draw_detections(frame, last_detections)
                writer.write(frame)
                task.processed = idx + 1
                idx += 1
        finally:
            cap.release()
            writer.release()
            task.video_path.unlink(missing_ok=True)  # 删除上传的临时视频

        if task.status == "cancelled":
            raw_output.unlink(missing_ok=True)
            return

        # 有 ffmpeg 则转 h264（浏览器/播放器兼容性更好），否则保留 mp4v
        task.output_path = OUTPUT_DIR / f"{task.id}.mp4"
        if shutil.which("ffmpeg"):
            ret = subprocess.run(  # noqa: S603 固定参数调用本机 ffmpeg
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_output),
                 "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(task.output_path)],
                capture_output=True, timeout=600,
            )
            raw_output.unlink(missing_ok=True)
            if ret.returncode != 0:
                raise RuntimeError(f"ffmpeg 转码失败: {ret.stderr.decode(errors='ignore')[:300]}")
        else:
            raw_output.rename(task.output_path)
