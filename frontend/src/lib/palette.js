/** 按类别名生成稳定的颜色（同类别同色，跨图一致）。 */
export function classColor(name) {
  let h = 0
  const s = String(name)
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.codePointAt(i)) % 360
  return `hsl(${h}, 78%, 58%)`
}

/** 带 alpha 的同色系。 */
export function classColorAlpha(name, alpha) {
  let h = 0
  const s = String(name)
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.codePointAt(i)) % 360
  return `hsla(${h}, 78%, 58%, ${alpha})`
}
