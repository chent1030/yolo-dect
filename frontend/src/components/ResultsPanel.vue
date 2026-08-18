<template>
  <div class="results-panel" v-if="result">
    <div class="summary">
      <span class="badge" :class="result.task">{{ result.task.toUpperCase() }}</span>
      <span class="model-name">{{ result.model_name }}</span>
      <span class="sep">·</span>
      <span>检出 <b>{{ visibleDetections.length }}</b>{{ visibleDetections.length !== result.detections.length ? ` / ${result.detections.length}` : '' }} 个目标</span>
      <span class="sep">·</span>
      <span class="mono">{{ result.inference_ms }} ms</span>
      <template v-if="fps">
        <span class="sep">·</span>
        <span class="mono">{{ fps.toFixed(1) }} fps</span>
      </template>
      <span class="sep">·</span>
      <span class="mono">{{ result.device.toUpperCase() }}</span>
      <span class="spacer"></span>
      <span class="filter-label">显示阈值</span>
      <input type="range" min="0" max="1" step="0.01" :value="displayConf" @input="$emit('update:displayConf', Number($event.target.value))" />
      <b class="mono conf-val">{{ displayConf.toFixed(2) }}</b>
      <button class="btn small" @click="$emit('export')">导出标注图</button>
      <button v-if="isVideo" class="btn small" :disabled="exportBusy" @click="$emit('export-video')">
        {{ exportBusy ? '导出中…' : '导出标注视频' }}
      </button>
    </div>

    <div v-if="exportTask" class="export-task" :class="exportTask.status">
      <template v-if="exportTask.status === 'running' || exportTask.status === 'pending'">
        <div class="progress"><div class="progress-fill" :style="{ width: exportTask.percent + '%' }"></div></div>
        <span>后端分析中 {{ exportTask.percent }}%（{{ exportTask.processed }}/{{ exportTask.total_frames }} 帧）</span>
        <button class="btn small" @click="$emit('cancel-export')">取消</button>
      </template>
      <template v-else-if="exportTask.status === 'done'">
        <span>✅ 标注视频已生成（{{ exportTask.detections_count }} 目标），已开始下载</span>
        <button class="btn small" @click="$emit('download-export', exportTask.id)">重新下载</button>
      </template>
      <template v-else-if="exportTask.status === 'cancelled'">
        <span>已取消导出</span>
      </template>
      <template v-else>
        <span class="err">导出失败: {{ exportTask.error || exportTask.status }}</span>
      </template>
    </div>

    <div class="chips" v-if="classStats.length">
      <button
        v-for="cs in classStats" :key="cs.name"
        class="chip" :class="{ off: hiddenClasses.has(cs.name) }"
        @click="$emit('toggle-class', cs.name)"
        :title="hiddenClasses.has(cs.name) ? '点击显示该类别' : '点击隐藏该类别'"
      >
        <i :style="{ background: classColor(cs.name) }"></i>
        {{ cs.name }} <b>{{ cs.visibleCount }}<span v-if="cs.visibleCount !== cs.count">/{{ cs.count }}</span></b>
      </button>
    </div>

    <div class="table-wrap" v-if="visibleDetections.length">
      <table>
        <thead>
          <tr><th style="width:36px">#</th><th>类别</th><th style="width:110px">得分</th><th style="width:200px">框 (x1 y1 x2 y2)</th></tr>
        </thead>
        <tbody>
          <tr
            v-for="(d, i) in visibleDetections" :key="i"
            :class="{ hover: i === hoverIdx }"
            @mouseenter="$emit('row-hover', i)"
            @mouseleave="$emit('row-hover', -1)"
          >
            <td class="mono dim">{{ i + 1 }}</td>
            <td><i class="dot" :style="{ background: classColor(d.class_name) }"></i>{{ d.class_name }}</td>
            <td>
              <div class="score-bar"><div class="fill" :style="{ width: (d.score * 100) + '%', background: classColor(d.class_name) }"></div></div>
              <span class="mono score-val">{{ d.score.toFixed(3) }}</span>
            </td>
            <td class="mono dim">{{ d.box.map((v) => Math.round(v)).join(' ') }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-else class="empty-hint">当前阈值/过滤条件下无可见目标</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { classColor } from '../lib/palette'

const props = defineProps({
  result: { type: Object, default: null },
  visibleDetections: { type: Array, default: () => [] },
  displayConf: { type: Number, default: 0 },
  hiddenClasses: { type: Set, default: () => new Set() },
  hoverIdx: { type: Number, default: -1 },
  isVideo: { type: Boolean, default: false },
  fps: { type: Number, default: 0 },
  exportTask: { type: Object, default: null },
  exportBusy: { type: Boolean, default: false },
})
defineEmits(['update:displayConf', 'toggle-class', 'row-hover', 'export', 'export-video', 'cancel-export', 'download-export'])

const classStats = computed(() => {
  if (!props.result) return []
  const map = new Map()
  for (const d of props.result.detections) {
    if (!map.has(d.class_name)) map.set(d.class_name, { name: d.class_name, count: 0, visibleCount: 0 })
    const cs = map.get(d.class_name)
    cs.count++
    if (!props.hiddenClasses.has(d.class_name) && d.score >= props.displayConf) cs.visibleCount++
  }
  return [...map.values()].sort((a, b) => b.count - a.count)
})
</script>

<style scoped>
.results-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  padding: 12px 14px;
}
.summary { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--text-dim); flex-wrap: wrap; }
.summary b { color: var(--text); }
.model-name { color: var(--text); font-family: var(--mono); font-size: 12px; }
.sep { color: var(--text-faint); }
.spacer { flex: 1; }
.mono { font-family: var(--mono); font-size: 12px; }
.conf-val { color: var(--accent); min-width: 32px; }
.filter-label { font-size: 12px; }
.summary input[type='range'] { width: 110px; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid var(--border-soft);
  background: var(--panel-2);
  color: var(--text);
  font-size: 12px;
  cursor: pointer;
}
.chip i { width: 9px; height: 9px; border-radius: 50%; }
.chip b { font-family: var(--mono); font-weight: 600; color: var(--text-dim); }
.chip.off { opacity: 0.4; }
.chip.off b { text-decoration: line-through; }

.table-wrap { max-height: 200px; overflow-y: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th {
  text-align: left;
  font-size: 11px;
  color: var(--text-faint);
  font-weight: 600;
  padding: 4px 8px;
  border-bottom: 1px solid var(--border-soft);
  position: sticky;
  top: 0;
  background: var(--panel);
}
td { padding: 4px 8px; border-bottom: 1px solid var(--border-soft); }
tr.hover td { background: rgba(61, 220, 151, 0.06); }
td .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }
.dim { color: var(--text-dim); }

.score-bar { display: inline-block; width: 60px; height: 5px; background: #1c2735; border-radius: 3px; overflow: hidden; vertical-align: middle; margin-right: 7px; }
.fill { height: 100%; border-radius: 3px; }
.score-val { color: var(--text-dim); vertical-align: middle; }

.empty-hint { padding: 18px; text-align: center; color: var(--text-faint); font-size: 13px; }

.export-task {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 8px;
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
  font-size: 12.5px;
  color: var(--text-dim);
}
.export-task .err { color: var(--red); }
.export-task .progress {
  flex: 1;
  min-width: 120px;
  height: 6px;
  border-radius: 3px;
  background: #1c2735;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 3px;
  transition: width 0.4s ease;
}
</style>
