async function parse(r) {
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail || `${r.status} ${r.statusText}`)
  return data
}

export async function fetchScenes() {
  const data = await parse(await fetch('/api/scenes'))
  return data.scenes || []
}

export async function fetchHealth() {
  return parse(await fetch('/api/health'))
}

export async function rescan() {
  return parse(await fetch('/api/admin/rescan', { method: 'POST' }))
}

export async function detect({ file, modelId, conf, iou, imgsz, maxDet }) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('model_id', modelId)
  if (conf != null) fd.append('conf', String(conf))
  if (iou != null) fd.append('iou', String(iou))
  if (imgsz != null) fd.append('imgsz', String(imgsz))
  if (maxDet != null) fd.append('max_det', String(maxDet))
  return parse(await fetch('/api/detect', { method: 'POST', body: fd }))
}
