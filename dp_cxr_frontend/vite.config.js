import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API calls are proxied rather than pointed at http://localhost:8000
// directly. Same-origin requests cannot trip CORS, which removes an entire
// class of "request failed" errors that have nothing to do with the model.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
    proxy: {
      '/health':       { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/predict':      { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/predict-json': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/model-card':   { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
