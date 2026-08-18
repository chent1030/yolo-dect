// overlay.js 绘制逻辑单测：mock canvas context，校验调用序列与坐标变换
import { drawOverlay, SKELETON } from '../src/lib/overlay.js'
import assert from 'node:assert'

const calls = []
const ctx = new Proxy(
  {},
  {
    get(_, prop) {
      if (prop === 'measureText') return (t) => ({ width: t.length * 6 })
      if (prop === 'setTransform' || prop === 'clearRect') return () => {}
      return (...args) => calls.push([prop, ...args])
    },
    set(_, prop, v) {
      calls.push(['set', prop, v])
      return true
    },
  }
)

const dets = [
  {
    box: [100, 50, 300, 400], score: 0.9, class_id: 0, class_name: 'person', mask: null,
    keypoints: Array.from({ length: 17 }, (_, i) => [100 + i * 5, 60 + i * 10, 0.9]),
  },
  {
    box: [10, 20, 60, 80], score: 0.7, class_id: 5, class_name: 'bus',
    mask: [[10, 20], [60, 20], [60, 80], [10, 80]], keypoints: null,
  },
]
drawOverlay(ctx, { detections: dets, scale: 0.5, dx: 10, dy: 20, hoverIdx: 0 })

const strokeRects = calls.filter((c) => c[0] === 'strokeRect')
const strokes = calls.filter((c) => c[0] === 'stroke')
const arcs = calls.filter((c) => c[0] === 'arc')

assert.equal(strokeRects.length, 2, '两个目标各画一个框')
const bus = strokeRects.find((c) => c[1] === 15) // [10,20,60,80]*0.5+(10,20) => x=15,y=30,w=25,h=30
assert.ok(bus, 'bus 框存在')
assert.deepEqual(bus.slice(1, 5), [15, 30, 25, 30], '坐标变换正确')
assert.ok(strokes.length > SKELETON.length, '骨架线 + 掩码轮廓均已绘制')
assert.equal(arcs.length, 17, '17 个关键点关节圆')
assert.ok(calls.some((c) => c[0] === 'set' && c[1] === 'shadowBlur'), 'hover 目标有高亮阴影')
assert.ok(calls.some((c) => c[0] === 'fillText' && String(c[1]).includes('bus 0.70')), '标签含类别与得分')
assert.ok(calls.some((c) => c[0] === 'roundRect'), '标签使用圆角底板')

console.log('overlay.test 全部通过: 框/掩码/骨架/标签/悬停高亮/坐标变换')
