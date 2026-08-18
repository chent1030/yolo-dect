"""视频标注渲染：用 OpenCV 在帧上绘制检测框/掩码/骨架。

配色算法与前端 palette.js 保持一致（同名类别同色）。
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

# COCO 17 关键点骨架连接（与前端 overlay.js 一致）
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
]
KEYPOINT_MIN_CONF = 0.5
MASK_ALPHA = 0.32


def class_color_bgr(name: str) -> tuple[int, int, int]:
    """与前端 classColor 相同的哈希 + HSL(78%,58%)，返回 BGR。"""
    h = 0
    for ch in str(name):
        h = (h * 31 + ord(ch)) % 360
    r, g, b = colorsys.hls_to_rgb(h / 360, 0.58, 0.78)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """在帧上绘制全部检测结果（返回原帧的标注副本）。"""
    img = frame

    # 掩码垫底：先收集到 overlay 层，一次性混合
    overlay = None
    for det in detections:
        mask = det.get("mask")
        if mask:
            if overlay is None:
                overlay = img.copy()
            pts = np.array(mask, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], class_color_bgr(det["class_name"]))
    if overlay is not None:
        cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0, img)

    # 检测框 + 标签
    for det in detections:
        box = det.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = class_color_bgr(det["class_name"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _draw_label(img, f"{det['class_name']} {det['score']:.2f}", x1, y1, color)

    # 骨架最上层
    for det in detections:
        kps = det.get("keypoints")
        if not kps:
            continue
        color = class_color_bgr(det["class_name"])
        for a, b in SKELETON:
            if kps[a][2] < KEYPOINT_MIN_CONF or kps[b][2] < KEYPOINT_MIN_CONF:
                continue
            pa = (int(round(kps[a][0])), int(round(kps[a][1])))
            pb = (int(round(kps[b][0])), int(round(kps[b][1])))
            cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
        for x, y, c in kps:
            if c < KEYPOINT_MIN_CONF:
                continue
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (255, 255, 255), -1)
            cv2.circle(img, (int(round(x)), int(round(y))), 4, color, 1, cv2.LINE_AA)
    return img


def _draw_label(img: np.ndarray, text: str, x1: int, y1: int, color) -> None:
    scale, thickness = 0.45, 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 2
    # 贴顶时标签放框内
    ly = y1 - th - baseline - 2 * pad
    if ly < 0:
        ly = y1 + 2
    lx = max(0, x1)
    cv2.rectangle(img, (lx, ly), (lx + tw + 2 * pad, ly + th + baseline + 2 * pad), color, -1)
    cv2.putText(
        img, text, (lx + pad, ly + th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale,
        (11, 15, 20), thickness, cv2.LINE_AA,
    )
