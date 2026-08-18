import { classColor, classColorAlpha } from './palette'

/** COCO 17 关键点骨架连接关系 */
export const SKELETON = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
  [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
]

const FONT = '600 11px ui-monospace, "SF Mono", Menlo, monospace'
const KEYPOINT_MIN_CONF = 0.5

/**
 * 在 canvas 上绘制检测结果。
 * @param ctx canvas 2d 上下文（已含 dpr 缩放）
 * @param opts { detections, scale, dx, dy, hoverIdx, task }
 *   scale/dx/dy: 图像坐标 -> canvas CSS 坐标的变换（先缩放后平移）
 */
export function drawOverlay(ctx, { detections, scale, dx, dy, hoverIdx }) {
  if (!detections) return

  // 掩码垫底
  for (const det of detections) {
    if (det.mask && det.mask.length > 2) drawMask(ctx, det, scale, dx, dy)
  }
  // 框 + 标签
  detections.forEach((det, i) => {
    if (!det.box) return
    drawBox(ctx, det, scale, dx, dy, i === hoverIdx)
  })
  // 骨架最上层
  for (const det of detections) {
    if (det.keypoints && det.keypoints.length) drawPose(ctx, det, scale, dx, dy)
  }
}

function toCanvas(pt, scale, dx, dy) {
  return [pt[0] * scale + dx, pt[1] * scale + dy]
}

function drawMask(ctx, det, scale, dx, dy) {
  ctx.beginPath()
  det.mask.forEach((pt, i) => {
    const [x, y] = toCanvas(pt, scale, dx, dy)
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  })
  ctx.closePath()
  ctx.fillStyle = classColorAlpha(det.class_name, 0.32)
  ctx.fill()
  ctx.strokeStyle = classColorAlpha(det.class_name, 0.9)
  ctx.lineWidth = 1.5
  ctx.stroke()
}

function drawBox(ctx, det, scale, dx, dy, hovered) {
  const [x1, y1, x2, y2] = det.box
  const x = x1 * scale + dx
  const y = y1 * scale + dy
  const w = (x2 - x1) * scale
  const h = (y2 - y1) * scale
  const color = classColor(det.class_name)

  ctx.save()
  if (hovered) {
    ctx.shadowColor = color
    ctx.shadowBlur = 10
  }
  ctx.strokeStyle = color
  ctx.lineWidth = hovered ? 3.5 : 2
  ctx.strokeRect(x, y, w, h)
  ctx.restore()

  // 标签: "name score"
  const text = `${det.class_name} ${det.score.toFixed(2)}`
  ctx.font = FONT
  const tw = ctx.measureText(text).width
  const chipH = 18
  const chipW = tw + 12
  let cy = y - chipH - 2
  if (cy < 2) cy = Math.min(y + 2, y + h - chipH) // 贴顶时放框内
  const cx = Math.max(2, Math.min(x, x + w - chipW))

  ctx.fillStyle = color
  ctx.beginPath()
  ctx.roundRect(cx, cy, chipW, chipH, 3)
  ctx.fill()
  ctx.fillStyle = '#0b0f14'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, cx + 6, cy + chipH / 2 + 0.5)
}

function drawPose(ctx, det, scale, dx, dy) {
  const kps = det.keypoints
  const color = classColor(det.class_name)
  const at = (i) => toCanvas(kps[i], scale, dx, dy)

  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.lineCap = 'round'
  for (const [a, b] of SKELETON) {
    if (kps[a][2] < KEYPOINT_MIN_CONF || kps[b][2] < KEYPOINT_MIN_CONF) continue
    const [ax, ay] = at(a)
    const [bx, by] = at(b)
    ctx.beginPath()
    ctx.moveTo(ax, ay)
    ctx.lineTo(bx, by)
    ctx.stroke()
  }
  for (let i = 0; i < kps.length; i++) {
    if (kps[i][2] < KEYPOINT_MIN_CONF) continue
    const [x, y] = at(i)
    ctx.beginPath()
    ctx.arc(x, y, 3.5, 0, Math.PI * 2)
    ctx.fillStyle = '#fff'
    ctx.fill()
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.stroke()
  }
}
