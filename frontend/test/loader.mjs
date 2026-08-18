// Node 原生 ESM 需要显式扩展名，此 loader 为无扩展名的相对导入补 .js（Vite 语义）
export async function resolve(specifier, context, next) {
  if (specifier.startsWith('.') && !/\.[a-z]+$/i.test(specifier)) {
    try {
      return await next(specifier + '.js', context)
    } catch {
      /* 回落到默认解析 */
    }
  }
  return next(specifier, context)
}
