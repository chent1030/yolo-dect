# YOLO 检测平台

后端丢权重即用、前端按场景调用的 YOLO 检测平台。左侧原始图片/视频、右侧检测结果（目标框 / 标签 / 得分），兼容 YOLOv5 ~ v12 的 det / seg / pose 三种任务，支持图片检测与视频检测（实时预览 + 导出标注视频）。

## 功能

- **即插即用**：把 `.pt` 权重放进 `backend/weights/<场景名>/`，后端 30 秒内（或点"刷新场景"）自动注册，前端立即可选
- **全版本兼容**：自动嗅探 checkpoint 选择引擎——ultralytics 引擎跑 YOLOv8/v9/v10/v11/v12，yolov5 引擎（torch.hub + 本地仓库）跑 YOLOv5
- **三种任务**：目标检测（框+标签+得分）、实例分割（半透明掩码）、姿态估计（COCO 17 点骨架）
- **图片检测**：前端 canvas 渲染，悬停高亮、显示阈值本地实时过滤、类别开关、导出标注图
- **视频检测（混合架构）**：
  - 实时预览：前端抽帧循环送检（带背压与帧率上限），播放中右侧实时出标注帧；暂停/拖动进度条精确检测当前帧
  - 导出标注视频：后端异步任务逐帧检测渲染（每 N 帧检测一次、中间帧复用结果），前端轮询进度，完成后自动下载 H.264 mp4，并附逐帧目标统计
- **工程化**：模型懒加载 + LRU 淘汰（防显存爆）、同模型推理串行锁、权重文件热更新（mtime 变化自动重载）、设备自动探测（cuda > mps > cpu）

## 快速开始

```bash
# 后端 (Python 3.13, uv 管理)
cd backend
uv sync                                   # 创建 venv 并安装依赖
uv run uvicorn app.main:app --port 8000   # 启动服务

# 前端 (Node 18+)
cd frontend
npm install
npm run dev        # http://localhost:5173，/api 已代理到 8000
```

打开 http://localhost:5173 ，选择场景和模型，拖入/粘贴/上传图片即可检测。

## 添加模型

```
backend/weights/
├── 安全帽检测/
│   ├── helmet_yolov8s.pt      # 任意版本 .pt，自动识别引擎/任务/类别
│   └── scene.yaml             # 可选：覆盖元信息
└── 火灾检测/
    └── fire_yolo11m.pt
```

可选的 `scene.yaml`：

```yaml
description: 工地安全帽检测
models:
  helmet_yolov8s.pt:
    description: 主模型
    conf: 0.35        # 该模型默认推理参数（可被前端逐次覆盖）
    imgsz: 960
```

场景目录里放一张 jpg/png 即成为"示例图片"按钮的数据源（见 demo 场景的 bus.jpg）。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/scenes` | 场景/模型清单（含任务类型、引擎、类别表） |
| POST | `/api/detect` | 图片检测：multipart `file` + `model_id` + 可选 `conf/iou/imgsz/max_det`（视频实时预览同样走此接口，前端逐帧调用） |
| POST | `/api/video/analyze` | 创建视频整段分析任务：multipart `file` + `model_id` + 参数 + `frame_step`（每 N 帧检测一次，中间帧复用结果） |
| GET | `/api/video/tasks/{id}` | 任务状态/进度/逐类统计 |
| POST | `/api/video/tasks/{id}/cancel` | 取消任务 |
| GET | `/api/video/tasks/{id}/download` | 下载标注视频 mp4（有 ffmpeg 时输出 H.264） |
| POST | `/api/admin/rescan` | 手动触发权重目录重扫描 |
| GET | `/api/sample-image` / `/api/sample-video` | 取场景目录里的示例图/视频 |
| GET | `/api/health` | 设备/场景/模型/已加载模型 |

统一检测结果 schema（三种任务一套）：

```json
{
  "model_id": "demo/yolov8n-seg.pt", "task": "segment", "device": "mps",
  "image": {"width": 810, "height": 1080}, "inference_ms": 157.9,
  "detections": [
    {"box": [670, 389, 810, 877], "score": 0.878, "class_id": 0, "class_name": "person",
     "mask": [[x, y], ...],          // 仅 segment
     "keypoints": [[x, y, conf], ...] // 仅 pose
    }
  ]
}
```

## 配置（backend/config.yaml）

| 键 | 默认 | 说明 |
|---|---|---|
| `device` | `auto` | `auto`/`cuda`/`mps`/`cpu`，auto 按 cuda>mps>cpu 探测 |
| `max_loaded_models` | `2` | 常驻模型数上限（LRU 淘汰），按显存调整 |
| `rescan_interval_sec` | `30` | 权重目录重扫描周期 |
| `half` | `false` | FP16 推理（仅 cuda） |
| `infer.*` | — | 推理参数全局默认值 |

## 部署到 NVIDIA 服务器

1. 服务器安装 [uv](https://docs.astral.sh/uv/)、CUDA 环境与 **ffmpeg**（标注视频转 H.264 用，无 ffmpeg 时输出 mp4v）
2. `cd backend && uv sync`，CUDA 版 torch 需按 [ultralytics 指南](https://docs.ultralytics.com/quickstart/) 选择对应索引源安装
3. `config.yaml` 设 `device: cuda`、`half: true`（可选），`max_loaded_models` 按显存加大
4. 前端 `npm run build` 后用任意静态服务器托管 `dist/`，并把 `/api` 反代到后端 8000 端口

## 兼容性说明

| 版本 | 引擎 | det | seg | pose |
|---|---|---|---|---|
| YOLOv8 / v9 / v10 / v11 / v12 | ultralytics | ✅ | ✅ | ✅ |
| YOLOv5 | yolov5 (third_party/yolov5) | ✅ | ✅ | ❌ 未验证 |

- v5 权重依赖 `backend/third_party/yolov5` 本地仓库（checkpoint pickle 引用其 `models` 包）
- 首次推理含模型加载耗时（数秒），后续同模型毫秒级；长时间不用的模型会被 LRU 卸载

## 项目结构

```
backend/
├── app/
│   ├── engines/          # 版本适配层：嗅探 + ultralytics/yolov5 双引擎
│   ├── registry.py       # weights/ 扫描、元信息缓存、scene.yaml 合并
│   ├── manager.py        # 懒加载、LRU、串行锁、热更新
│   ├── video_annotator.py # 视频帧标注渲染（cv2，配色与前端一致）
│   ├── video_tasks.py    # 视频分析任务（后台线程/进度/取消/导出）
│   ├── api.py            # 路由
│   └── main.py           # 入口 + 定时重扫描
├── weights/              # 权重仓库（场景=目录，可放示例图/视频）
├── outputs/              # 标注视频输出（自动生成）
├── third_party/yolov5/   # v5 引擎依赖
└── config.yaml
frontend/
├── src/
│   ├── App.vue           # 主布局、图片/视频双模式状态机
│   ├── components/       # ControlPanel / ImageCanvas / ResultsPanel
│   ├── lib/overlay.js    # canvas 绘制（框/掩码/骨架/标签）
│   └── lib/palette.js    # 类别配色
└── test/                 # overlay 绘制单测：node --import ./test/register.mjs test/overlay.test.mjs
```
