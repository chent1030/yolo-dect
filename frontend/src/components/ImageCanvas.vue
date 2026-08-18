<template>
  <div ref="wrapEl" class="image-canvas" :class="{ dropping }" @dragover.prevent="dropping = true" @dragleave="dropping = false" @drop.prevent="onDrop">
    <canvas ref="canvasEl" @mousemove="onMove" @mouseleave="onLeave"></canvas>
    <div v-if="!imageUrl" class="placeholder">
      <div class="ph-icon">{{ icon }}</div>
      <div class="ph-title">{{ title }}</div>
      <div class="ph-sub">{{ subtitle }}</div>
      <slot name="actions"></slot>
    </div>
    <div v-if="loading" class="loading-mask">
      <div class="spinner"></div>
      <div>{{ loadingText }}</div>
    </div>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { drawOverlay } from '../lib/overlay'

const props = defineProps({
  imageUrl: { type: String, default: null },
  detections: { type: Array, default: () => [] },
  hoverIdx: { type: Number, default: -1 },
  loading: { type: Boolean, default: false },
  loadingText: { type: String, default: '推理中…' },
  icon: { type: String, default: '🖼️' },
  title: { type: String, default: '拖入图片到此处' },
  subtitle: { type: String, default: '' },
})
const emit = defineEmits(['hover', 'file-dropped'])

const wrapEl = ref(null)
const canvasEl = ref(null)
const dropping = ref(false)

const imgEl = shallowRef(null)
// 图像坐标 -> canvas CSS 坐标变换，hover 命中检测用
let transform = { scale: 1, dx: 0, dy: 0 }

function load() {
  if (!props.imageUrl) { imgEl.value = null; redraw(); return }
  const im = new Image()
  // 注意：加载新图期间保留旧图显示，避免视频高频换帧时闪烁
  im.onload = () => { imgEl.value = im; redraw() }
  im.src = props.imageUrl
}

function redraw() {
  const canvas = canvasEl.value
  const wrap = wrapEl.value
  if (!canvas || !wrap) return
  const dpr = window.devicePixelRatio || 1
  const cw = wrap.clientWidth
  const ch = wrap.clientHeight
  if (!cw || !ch) return
  canvas.width = cw * dpr
  canvas.height = ch * dpr
  canvas.style.width = cw + 'px'
  canvas.style.height = ch + 'px'
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cw, ch)

  const im = imgEl.value
  if (!im) return
  const scale = Math.min(cw / im.naturalWidth, ch / im.naturalHeight, 1)
  const dw = im.naturalWidth * scale
  const dh = im.naturalHeight * scale
  const dx = (cw - dw) / 2
  const dy = (ch - dh) / 2
  transform = { scale, dx, dy }
  ctx.drawImage(im, dx, dy, dw, dh)
  drawOverlay(ctx, {
    detections: props.detections,
    scale, dx, dy,
    hoverIdx: props.hoverIdx,
  })
}

function onMove(e) {
  const canvas = canvasEl.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top
  const { scale, dx, dy } = transform
  const ix = (mx - dx) / scale
  const iy = (my - dy) / scale
  // 顶层优先（后画的在上面）
  for (let i = props.detections.length - 1; i >= 0; i--) {
    const [x1, y1, x2, y2] = props.detections[i].box || []
    if (ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2) { emit('hover', i); return }
  }
  emit('hover', -1)
}

function onLeave() { emit('hover', -1) }

function onDrop(e) {
  dropping.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file && (file.type.startsWith('image/') || file.type.startsWith('video/'))) emit('file-dropped', file)
}

defineExpose({
  exportDataURL: () => canvasEl.value?.toDataURL('image/png') || null,
})

watch(() => props.imageUrl, load)
watch(() => [props.detections, props.hoverIdx], redraw, { deep: false })

let ro = null
onMounted(() => {
  ro = new ResizeObserver(redraw)
  ro.observe(wrapEl.value)
  load()
})
onBeforeUnmount(() => ro?.disconnect())
</script>

<style scoped>
.image-canvas {
  position: relative;
  width: 100%;
  height: 100%;
  border-radius: var(--radius);
  background:
    linear-gradient(45deg, #10161f 25%, transparent 25%),
    linear-gradient(-45deg, #10161f 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #10161f 75%),
    linear-gradient(-45deg, transparent 75%, #10161f 75%);
  background-size: 16px 16px;
  background-position: 0 0, 0 8px, 8px -8px, -8px 0;
  overflow: hidden;
}
.image-canvas.dropping { outline: 2px dashed var(--accent); outline-offset: -4px; }
canvas { display: block; }

.placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--text-faint);
  pointer-events: none;
}
.ph-icon { font-size: 40px; opacity: 0.6; }
.ph-title { font-size: 15px; color: var(--text-dim); }
.ph-sub { font-size: 12px; }
.placeholder :deep(.btn) { pointer-events: auto; margin-top: 6px; }

.loading-mask {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: rgba(11, 15, 20, 0.55);
  backdrop-filter: blur(2px);
  color: var(--text);
  font-size: 13px;
}
.spinner {
  width: 28px;
  height: 28px;
  border: 3px solid rgba(61, 220, 151, 0.25);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
