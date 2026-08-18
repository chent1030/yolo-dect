"""FastAPI 应用入口。启动: uv run uvicorn app.main:app --port 8000"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .config import load_config
from .manager import ModelManager
from .video_tasks import VideoTaskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    cfg = load_config()
    manager = ModelManager(cfg)
    manager.rescan()
    logger.info(
        "服务启动: device=%s, 已注册 %d 场景 / %d 模型, 权重目录=%s",
        manager.device,
        len(manager.scenes),
        sum(len(s.models) for s in manager.scenes.values()),
        cfg.weights_dir,
    )

    stop = threading.Event()

    def _rescan_loop() -> None:
        while not stop.wait(cfg.rescan_interval_sec):
            try:
                manager.rescan()
            except Exception:
                logger.exception("定时重扫描失败")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        thread = threading.Thread(target=_rescan_loop, daemon=True, name="rescan")
        thread.start()
        yield
        stop.set()

    app = FastAPI(title="YOLO 检测服务", version="1.1.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.video_tasks = VideoTaskManager(manager)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 内部工具，跨域放开；生产按需收紧
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
