import fs from "fs"
import path from "path"
import { defineConfig, type Plugin } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// Cloudflare Pages serves public/how/index.html at /how/; the dev server
// would hand /how to the SPA instead. Match production.
const howPage = (): Plugin => ({
  name: "how-page",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const [p, q] = (req.url || "").split("?")
      if (p === "/how") {
        res.writeHead(301, { Location: "/how/" + (q ? "?" + q : "") }).end()
      } else if (p === "/how/") {
        res.setHeader("Content-Type", "text/html; charset=utf-8")
        res.end(fs.readFileSync(path.resolve(__dirname, "public/how/index.html")))
      } else next()
    })
  },
})

export default defineConfig({
  plugins: [howPage(), react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  // Dev hits the configured API origin, so streaming
  // behaviour is identical in both.
  server: {
    proxy: {
      "/api": {
        target: process.env.PAL_API || "https://palindrome-api.ericspencer.us",
        changeOrigin: true, secure: true,
      },
    },
  },
})
