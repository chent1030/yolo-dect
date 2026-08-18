<template>
  <div class="layout">
    <header class="topbar">
      <div class="brand">
        <span class="logo"></span>
        <h1>YOLO 检测平台</h1>
        <span class="ver">v5 – v12 · det / seg / pose · 图片 / 视频</span>
      </div>
      <div class="top-right">
        <span v-if="health" class="device-chip mono" :class="health.device">{{ health.device.toUpperCase() }}</span>
        <span class="top-info">{{ health ? `${health.scenes} 场景 · ${health.models} 模型` : '后端未连接' }}</span>
      </div>
    </header>

    <div v-if="error" class="error-bar">
      <span>{{ error }}</span>
      <button class="close" @click="error = ''">×</button>
    </div>

    <div class="body">
      <aside class="sidebar">
        <ControlPanel
          v-model:mode="mode"
          v-model:scene="sceneName"
          v-model:modelId="modelId"
          v-model:conf="params.conf"
          v-model:iou="params.iou"
          v-model:imgsz="params.imgsz"
          v-model:maxDet="params.maxDet"
          v-model:maxFps="maxFps"
          v-model:frameStep="frameStep"
          :scenes="scenes"
          :has-media="mode === 'image' ? !!imageUrl : !!videoUrl"
          :loading="loading"
          @run="mode === 'image' ? runDetect() : detectFrame()"
          @pick-media="pickMedia"
          @rescan="doRescan"
        />
      </aside>

      <main class="main">
        <div class="viewer-grid">
          <div class="viewer-card">
            <div class="viewer-head">
              <span class="viewer-title">{{ mode === 'image' ? '原始图片' : '原始视频' }}</span>
              <span v-if="mediaName" class="mono dim">{{ mediaName }}</span>
            </div>
            <div class="viewer-body" @dragover.prevent @drop.prevent="onDropLeft">
              <!-- 图片模式 -->
              <div v-show="mode === 'image'" class="plain-image">
                <img v-if="imageUrl" :src="imageUrl" alt="原始图片" />
                <div v-else class="placeholder">
                  <div class="ph-icon">🖼️</div>
                  <div class="ph-title">等待图片</div>
                  <div class="ph-sub">拖入 / 粘贴 (⌘V) / 左侧按钮上传</div>
                  <button class="btn small" @click="loadSample('image')">加载示例图片</button>
                </div>
              </div>
              <!-- 视频模式 -->
              <div v-show="mode === 'video'" class="plain-image">
                <video
                  v-if="videoUrl"
                  ref="videoEl"
                  :src="videoUrl"
                  controls
                  muted
                  loop
                  playsinline
                  autoplay
                  class="video-el"
                  @loadeddata="onVideoReady"
                  @play="startLoop"
                  @pause="detectPausedFrame"
                  @seeked="detectPausedFrame"
                ></video>
                <div v-else class="placeholder">
                  <div class="ph-icon">🎬</div>
                  <div class="ph-title">等待视频</div>
                  <div class="ph-sub">拖入视频 / 左侧按钮选择（mp4/webm/mov）</div>
                  <button class="btn small" @click="loadSample('video')">加载示例视频</button>
                </div>
              </div>
            </div>
          </div>

          <div class="viewer-card">
            <div class="viewer-head">
              <span class="viewer-title">{{ mode === 'image' ? '检测结果' : '检测预览' }}</span>
              <span v-if="activeResult" class="mono dim">{{ activeResult.image.width }}×{{ activeResult.image.height }}</span>
              <span v-else-if="!hasMedia" class="dim">未开始</span>
            </div>
            <div class="viewer-body">
              <ImageCanvas
                ref="canvasRef"
                :image-url="mode === 'image' ? imageUrl : (videoResult?.frameURL ?? null)"
                :detections="visibleDetections"
                :hover-idx="hoverIdx"
                :loading="rightLoading"
                :loading-text="mode === 'video' ? '视频检测启动中…' : '推理中…'"
                :icon="mode === 'video' ? '🎬' : '🖼️'"
                :title="mode === 'video' ? '等待视频检测' : '拖入图片到此处'"
                :subtitle="mode === 'video' ? '播放后自动逐帧检测右侧显示' : ''"
                @hover="hoverIdx = $event"
                @file-dropped="setMedia"
              />
            </div>
          </div>
        </div>

        <ResultsPanel
          v-model:display-conf="displayConf"
          :result="activeResult"
          :visible-detections="visibleDetections"
          :hidden-classes="hiddenClasses"
          :hover-idx="hoverIdx"
          :is-video="mode === 'video'"
          :fps="mode === 'video' ? fpsEma : 0"
          :export-task="exportTask"
          :export-busy="exportBusy"
          @toggle-class="toggleClass"
          @row-hover="hoverIdx = $event"
          @export="exportImage"
          @export-video="exportVideo"
          @cancel-export="cancelExport"
          @download-export="downloadExported"
        />
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import ControlPanel from './components/ControlPanel.vue'
import ImageCanvas from './components/ImageCanvas.vue'
import ResultsPanel from './components/ResultsPanel.vue'
import { detect, fetchHealth, fetchScenes, rescan as apiRescan } from './api'

const scenes = ref([])
const health = ref(null)
const sceneName = ref('')
const modelId = ref('')
const params = reactive({ conf: 0.25, iou: 0.7, imgsz: 640, maxDet: 300 })

const mode = ref('image')

// ---- 图片状态 ----
const imageFile = ref(null)
const imageUrl = ref(null)
const imageName = ref('')
const result = ref(null)
const loading = ref(false)

// ---- 视频状态 ----
const videoFile = ref(null)
const videoUrl = ref(null)
const videoName = ref('')
const videoResult = ref(null) // { ...detectResponse, frameURL }
const videoEl = ref(null)
const maxFps = ref(10)
const frameStep = ref(1)
const fpsEma = ref(0)
let inFlight = false
let pendingSeek = false
let lastShot = 0
let rafId = 0
const captureCanvas = document.createElement('canvas')

// ---- 共享 UI 状态 ----
const error = ref('')
const displayConf = ref(0)
const hiddenClasses = ref(new Set())
const hoverIdx = ref(-1)
const canvasRef = ref(null)

// ---- 导出标注视频任务 ----
const exportTask = ref(null)
const exportBusy = ref(false)
let pollTimer = 0

const hasMedia = computed(() => (mode.value === 'image' ? !!imageUrl.value : !!videoUrl.value))
const mediaName = computed(() => (mode.value === 'image' ? imageName.value : videoName.value))
const activeResult = computed(() => (mode.value === 'image' ? result.value : videoResult.value))
const rightLoading = computed(() =>
  mode.value === 'image' ? loading.value : (!!videoUrl.value && !videoResult.value)
)

const visibleDetections = computed(() => {
  if (!activeResult.value) return []
  return activeResult.value.detections.filter(
    (d) => d.score >= displayConf.value && !hiddenClasses.value.has(d.class_name)
  )
})

async function loadScenes(keepSelection = true) {
  try {
    scenes.value = await fetchScenes()
    health.value = await fetchHealth()
    if (!scenes.value.length) return
    if (!keepSelection || !scenes.value.some((s) => s.name === sceneName.value)) {
      sceneName.value = scenes.value[0].name
    }
  } catch (e) {
    error.value = `加载场景失败: ${e.message}（确认后端已启动在 8000 端口）`
  }
}

watch(sceneName, (name) => {
  const scene = scenes.value.find((s) => s.name === name)
  modelId.value = scene?.models[0]?.id || ''
  const meta = scene?.models[0]
  if (meta?.defaults) {
    if (meta.defaults.conf != null) params.conf = meta.defaults.conf
    if (meta.defaults.iou != null) params.iou = meta.defaults.iou
    if (meta.defaults.imgsz != null) params.imgsz = meta.defaults.imgsz
    if (meta.defaults.max_det != null) params.maxDet = meta.defaults.max_det
  }
})

// 换模型：图片重推理；视频暂停态补检当前帧（播放中由循环自动带上新模型）
watch(modelId, () => {
  if (mode.value === 'image' && imageFile.value && !loading.value) runDetect()
  else if (mode.value === 'video' && videoUrl.value && !inFlight) detectFrame()
})

// 离开视频模式时暂停播放（检测循环随暂停自动停止）
watch(mode, (m) => {
  if (m !== 'video' && videoEl.value) videoEl.value.pause()
})

function pickMedia() {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = mode.value === 'video' ? 'video/*' : 'image/*'
  input.onchange = () => { if (input.files?.[0]) setMedia(input.files[0]) }
  input.click()
}

function onDropLeft(e) {
  const file = e.dataTransfer?.files?.[0]
  if (file) setMedia(file)
}

function setMedia(file) {
  if (file.type.startsWith('video/')) {
    mode.value = 'video'
    if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
    videoFile.value = file
    videoUrl.value = URL.createObjectURL(file)
    videoName.value = file.name
    videoResult.value = null
    fpsEma.value = 0
    hoverIdx.value = -1
    hiddenClasses.value = new Set()
    error.value = ''
    exportTask.value = null
    // loadeddata 事件触发首轮检测，autoplay 播放后循环启动
  } else if (file.type.startsWith('image/')) {
    mode.value = 'image'
    if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
    imageFile.value = file
    imageUrl.value = URL.createObjectURL(file)
    imageName.value = file.name
    result.value = null
    hoverIdx.value = -1
    hiddenClasses.value = new Set()
    error.value = ''
    if (modelId.value) runDetect()
  } else {
    error.value = '仅支持图片或视频文件'
  }
}

async function loadSample(kind) {
  try {
    const r = await fetch(`/api/sample-${kind}`)
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText)
    const blob = await r.blob()
    setMedia(new File([blob], `sample.${kind === 'video' ? 'mp4' : 'jpg'}`, { type: blob.type }))
  } catch (e) {
    error.value = `加载示例失败: ${e.message}`
  }
}

// ---- 图片检测 ----

async function runDetect() {
  if (!imageFile.value || !modelId.value) return
  loading.value = true
  error.value = ''
  hoverIdx.value = -1
  try {
    result.value = await detect({
      file: imageFile.value,
      modelId: modelId.value,
      conf: params.conf, iou: params.iou, imgsz: params.imgsz, maxDet: params.maxDet,
    })
    displayConf.value = 0
  } catch (e) {
    error.value = `检测失败: ${e.message}`
  } finally {
    loading.value = false
  }
}

// ---- 视频实时预览：抽帧 → 送检 → 右侧画标注帧 ----

function captureFrame(video) {
  captureCanvas.width = video.videoWidth
  captureCanvas.height = video.videoHeight
  captureCanvas.getContext('2d').drawImage(video, 0, 0)
  return captureCanvas.toDataURL('image/jpeg', 0.85)
}

function dataURLtoFile(dataURL, name) {
  const [meta, b64] = dataURL.split(',')
  const mime = meta.match(/:(.*?);/)[1]
  const bin = atob(b64)
  const buf = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i)
  return new File([buf], name, { type: mime })
}

async function detectFrame() {
  const v = videoEl.value
  if (mode.value !== 'video' || !v || !v.videoWidth || !modelId.value || inFlight) return
  inFlight = true
  try {
    const frameURL = captureFrame(v)
    const res = await detect({
      file: dataURLtoFile(frameURL, 'frame.jpg'),
      modelId: modelId.value,
      conf: params.conf, iou: params.iou, imgsz: params.imgsz, maxDet: params.maxDet,
    })
    videoResult.value = { ...res, frameURL }
    error.value = ''
  } catch (e) {
    error.value = `视频检测失败: ${e.message}`
  } finally {
    inFlight = false
    if (pendingSeek) { pendingSeek = false; detectFrame() }
  }
}

function tick() {
  rafId = 0
  const v = videoEl.value
  if (mode.value !== 'video' || !v || v.paused || v.ended) return
  const now = performance.now()
  if (!inFlight && now - lastShot >= 1000 / maxFps.value) {
    // fps 按实际抽帧间隔统计（受帧率上限约束）
    if (lastShot) {
      const inst = 1000 / (now - lastShot)
      fpsEma.value = fpsEma.value ? fpsEma.value * 0.7 + inst * 0.3 : inst
    }
    lastShot = now
    detectFrame()
  }
  rafId = requestAnimationFrame(tick)
}

function startLoop() {
  if (!rafId) rafId = requestAnimationFrame(tick)
}

function onVideoReady() { detectFrame() }

// 暂停/拖动进度条后，精确检测当前停留帧
function detectPausedFrame() {
  const v = videoEl.value
  if (v && !v.paused) return // 播放中由循环覆盖
  if (inFlight) pendingSeek = true
  else detectFrame()
}

// ---- 导出标注视频（后端整段分析） ----

async function exportVideo() {
  if (!videoFile.value || !modelId.value) return
  exportBusy.value = true
  error.value = ''
  try {
    const fd = new FormData()
    fd.append('file', videoFile.value)
    fd.append('model_id', modelId.value)
    fd.append('conf', String(params.conf))
    fd.append('iou', String(params.iou))
    fd.append('imgsz', String(params.imgsz))
    fd.append('max_det', String(params.maxDet))
    fd.append('frame_step', String(frameStep.value))
    const r = await fetch('/api/video/analyze', { method: 'POST', body: fd })
    const data = await r.json()
    if (!r.ok) throw new Error(data.detail || r.statusText)
    exportTask.value = data
    startPolling(data.id)
  } catch (e) {
    error.value = `创建导出任务失败: ${e.message}`
  } finally {
    exportBusy.value = false
  }
}

function startPolling(id) {
  clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/video/tasks/${id}`)
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || r.statusText)
      exportTask.value = data
      if (data.status === 'done') { clearInterval(pollTimer); downloadExported(id) }
      else if (data.status === 'error' || data.status === 'cancelled') clearInterval(pollTimer)
    } catch (e) {
      clearInterval(pollTimer)
      exportTask.value = { ...(exportTask.value || { id }), status: 'error', error: e.message }
    }
  }, 1500)
}

function cancelExport() {
  if (exportTask.value?.id) {
    fetch(`/api/video/tasks/${exportTask.value.id}/cancel`, { method: 'POST' }).catch(() => {})
  }
}

function downloadExported(id) {
  const a = document.createElement('a')
  a.href = `/api/video/tasks/${id}/download`
  a.download = ''
  a.click()
}

// ---- 通用 ----

async function doRescan() {
  try {
    await apiRescan()
    await loadScenes()
  } catch (e) {
    error.value = `刷新失败: ${e.message}`
  }
}

function toggleClass(name) {
  const s = new Set(hiddenClasses.value)
  s.has(name) ? s.delete(name) : s.add(name)
  hiddenClasses.value = s
}

function exportImage() {
  const url = canvasRef.value?.exportDataURL()
  if (!url) return
  const a = document.createElement('a')
  const base = (mediaName.value || 'image').replace(/\.[^.]+$/, '')
  a.href = url
  a.download = `${base}_detected.png`
  a.click()
}

function onPaste(e) {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'))
  if (item) setMedia(item.getAsFile())
}

onMounted(() => {
  loadScenes(false)
  window.addEventListener('paste', onPaste)
})
onUnmounted(() => {
  window.removeEventListener('paste', onPaste)
  cancelAnimationFrame(rafId)
  clearInterval(pollTimer)
})
</script>

<style scoped>
.layout { display: flex; flex-direction: column; height: 100%; }

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 52px;
  border-bottom: 1px solid var(--border-soft);
  background: var(--panel-2);
  flex-shrink: 0;
}
.brand { display: flex; align-items: center; gap: 10px; }
.logo {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 10px rgba(61, 220, 151, 0.8);
}
.brand h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: 0.02em; }
.ver { font-size: 11px; color: var(--text-faint); font-family: var(--mono); }
.top-right { display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--text-dim); }
.device-chip {
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  border: 1px solid var(--border);
}
.device-chip.cuda { color: var(--accent); border-color: var(--accent-dim); }
.device-chip.mps { color: var(--blue); border-color: rgba(77, 163, 255, 0.4); }
.device-chip.cpu { color: var(--text-dim); }

.error-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 20px;
  background: rgba(255, 107, 107, 0.12);
  border-bottom: 1px solid rgba(255, 107, 107, 0.3);
  color: #ff9d9d;
  font-size: 13px;
}
.error-bar .close { background: none; border: none; color: #ff9d9d; font-size: 16px; cursor: pointer; }

.body { display: flex; flex: 1; min-height: 0; }
.sidebar {
  width: 280px;
  flex-shrink: 0;
  border-right: 1px solid var(--border-soft);
  padding: 14px;
  overflow-y: auto;
}

.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px 16px;
  min-width: 0;
  overflow-y: auto;
}

.viewer-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  min-height: 380px;
  flex: 1;
}
.viewer-card {
  display: flex;
  flex-direction: column;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  overflow: hidden;
  min-height: 0;
}
.viewer-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 13px;
  border-bottom: 1px solid var(--border-soft);
  flex-shrink: 0;
}
.viewer-title { font-size: 12.5px; font-weight: 600; color: var(--text-dim); letter-spacing: 0.04em; }
.viewer-body { flex: 1; min-height: 0; padding: 10px; }

.plain-image {
  position: relative;
  width: 100%;
  height: 100%;
  border-radius: var(--radius);
  overflow: hidden;
  background: repeating-conic-gradient(#10161f 0% 25%, #0d1219 0% 50%) 0 0 / 16px 16px;
}
.plain-image img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.video-el {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
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
.placeholder .btn { pointer-events: auto; margin-top: 6px; }

.mono { font-family: var(--mono); font-size: 12px; }
.dim { color: var(--text-faint); }
</style>
