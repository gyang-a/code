---
name: vite-network-config
description: 在 Vite 项目中配置开发服务器监听所有网络接口（0.0.0.0），实现局域网或容器内访问。
---

# Vite 网络配置 — 监听全局端口

## 问题

Vite 默认 `npm run dev` 仅监听 `localhost`，导致通过局域网 IP 或容器端口映射无法访问，出现连接拒绝错误。

## 解决方案

在 `vite.config.ts` 中添加 `server.host: '0.0.0.0'`。

```ts
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,  // 可根据需要修改
  },
})
```

## 触发条件

- 新建 Vite + React（或其他框架）项目时
- `vite.config.ts` 中缺少 `server.host` 配置
- 需要从外部网络或容器内部访问开发服务器

## 检查清单

- [ ] `vite.config.ts` 中是否包含 `server.host: '0.0.0.0'`？
- [ ] 端口 `server.port` 是否与预期一致（默认 5173）？
- [ ] 重新运行 `npm run dev` 后，控制台是否显示 `Local:   http://localhost:5173/` 以及 `Network: http://<IP>:5173/`？

## 注意事项

- **务必保留 `server.host: '0.0.0.0'`**，否则开发服务器无法监听所有网络接口。
- 如果端口被占用，Vite 会自动递增端口，但 `host` 配置依然生效。
- 生产构建（`vite build`）不受此配置影响。
- 此配置适用于所有 Vite 项目，不限于前端框架。
