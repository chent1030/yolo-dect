<template>
  <div class="control-panel">
    <div class="mode-switch">
      <button :class="{ active: mode === 'image' }" @click="$emit('update:mode', 'image')">🖼️ 图片检测</button>
      <button :class="{ active: mode === 'video' }" @click="$emit('update:mode', 'video')">🎬 视频检测</button>
    </div>

    <section class="group">
      <h3>场景与模型</h3>
      <label class="field">
        <span>场景</span>
        <select v-model="sceneName">
          <option v-if="!scenes.length" value="" disabled>暂无场景（后端 weights/ 下建目录放入 .pt）</option>
          <option v-for="s in scenes" :key="s.name" :value="s.name">
            {{ s.name }}{{ s.description ? ` · ${s.description}` : '' }}
          </option>
        </select>
      </label>
      <label class="field">
        <span>模型</span>
        <select v-model="modelId">
          <option v-if="!models.length" value="" disabled>该场景暂无模型</option>
          <option v-for="m in models" :key="m.id" :value="m.id">
            {{ m.filename }} ({{ m.size_mb }}MB)
          </option>
        </select>
      </label>
      <div v-if="model" class="model-meta">
        <span class="badge" :class="model.task">{{ model.task.toUpperCase() }}</span>
        <span class="badge engine">{{ model.engine === 'yolov5' ? 'YOLOv5' : 'Ultralytics' }}</span>
        <span class="meta-text">{{ model.classes.length }} 类</span>
        <div v-if="model.description" class="meta-desc">{{ model.description }}</div>
      </div>
    </section>

    <section class="group">
      <h3>推理参数</h3>
      <div class="field slider-field">
        <div class="slider-head"><span>置信度 conf</span><b>{{ conf.toFixed(2) }}</b></div>
        <input type="range" min="0.01" max="0.95" step="0.01" v-model.number="conf" />
      </div>
      <div class="field slider-field">
        <div class="slider-head"><span>NMS IoU</span><b>{{ iou.toFixed(2) }}</b></div>
        <input type="range" min="0.1" max="1" step="0.05" v-model.number="iou" />
      </div>
      <div class="field-row">
        <label class="field">
          <span>推理尺寸</span>
          <select v-model.number="imgsz">
            <option v-for="s in [320, 416, 512, 640, 800, 960, 1280]" :key="s" :value="s">{{ s }}px</option>
          </select>
        </label>
        <label class="field">
          <span>最大检测数</span>
          <input type="number" min="1" max="1000" v-model.number="maxDet" />
        </label>
      </div>
    </section>

    <section v-if="mode === 'video'" class="group">
      <h3>视频设置</h3>
      <div class="field slider-field">
        <div class="slider-head"><span>预览检测帧率上限</span><b>{{ maxFps }} fps</b></div>
        <input type="range" min="1" max="30" step="1" v-model.number="maxFps" />
      </div>
      <label class="field">
        <span>导出检测间隔</span>
        <select v-model.number="frameStep">
          <option :value="1">逐帧检测（最准）</option>
          <option :value="2">每 2 帧检测一次</option>
          <option :value="3">每 3 帧检测一次</option>
          <option :value="5">每 5 帧检测一次（最快）</option>
        </select>
      </label>
    </section>

    <section class="group actions">
      <button class="btn primary" :disabled="!modelId || !hasMedia || loading" @click="$emit('run')">
        {{ mode === 'video' ? '检测当前帧' : (loading ? '推理中…' : '开始检测') }}
      </button>
      <button class="btn" @click="$emit('pick-media')">
        {{ mode === 'video' ? '选择视频' : '选择图片' }}
      </button>
      <button class="btn" @click="$emit('rescan')" title="重新扫描后端权重目录">刷新场景</button>
    </section>

    <section v-if="model" class="group">
      <h3>类别 ({{ model.classes.length }})</h3>
      <div class="class-list">
        <span v-for="c in model.classes" :key="c" class="class-chip">
          <i :style="{ background: classColor(c) }"></i>{{ c }}
        </span>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { classColor } from '../lib/palette'

const props = defineProps({
  scenes: { type: Array, default: () => [] },
  hasMedia: Boolean,
  loading: Boolean,
})
const emit = defineEmits(['run', 'pick-media', 'rescan', 'update:mode'])

const mode = defineModel('mode', { type: String, default: 'image' })
const sceneName = defineModel('scene', { type: String, default: '' })
const modelId = defineModel('modelId', { type: String, default: '' })
const conf = defineModel('conf', { type: Number, default: 0.25 })
const iou = defineModel('iou', { type: Number, default: 0.7 })
const imgsz = defineModel('imgsz', { type: Number, default: 640 })
const maxDet = defineModel('maxDet', { type: Number, default: 300 })
const maxFps = defineModel('maxFps', { type: Number, default: 10 })
const frameStep = defineModel('frameStep', { type: Number, default: 1 })

const models = computed(() => props.scenes.find((s) => s.name === sceneName.value)?.models || [])
const model = computed(() => models.value.find((m) => m.id === modelId.value))
</script>

<style scoped>
.control-panel { display: flex; flex-direction: column; gap: 14px; }

.mode-switch {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
}
.mode-switch button {
  flex: 1;
  padding: 8px 0;
  border: none;
  border-radius: 7px;
  background: transparent;
  color: var(--text-dim);
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.mode-switch button.active {
  background: var(--accent);
  color: #06281a;
  font-weight: 600;
}

.group {
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  padding: 13px 14px;
}
h3 {
  margin: 0 0 11px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  color: var(--text-faint);
  text-transform: uppercase;
}
.field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }
.field > span { font-size: 12px; color: var(--text-dim); }
.field-row { display: flex; gap: 10px; }
.field-row .field { flex: 1; }

.slider-field .slider-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 12px;
  color: var(--text-dim);
}
.slider-head b { font-family: var(--mono); color: var(--accent); font-size: 12px; }

.model-meta { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.meta-text { font-size: 12px; color: var(--text-dim); }
.meta-desc { width: 100%; font-size: 12px; color: var(--text-dim); margin-top: 4px; }

.actions { display: flex; flex-direction: column; gap: 8px; }

.class-list { display: flex; flex-wrap: wrap; gap: 5px; max-height: 180px; overflow-y: auto; }
.class-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  font-size: 11px;
  color: var(--text-dim);
}
.class-chip i { width: 8px; height: 8px; border-radius: 50%; }
</style>
