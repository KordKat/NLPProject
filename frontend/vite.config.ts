import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  const target = env.VITE_API_PROXY || "http://localhost:8000";
  // hosts that may serve the dev server, e.g. a RunPod HTTPS proxy (<pod>-5173.proxy.runpod.net)
  const allowedHosts = ["localhost", "127.0.0.1", ".proxy.runpod.net", ...(env.VITE_ALLOWED_HOSTS || "").split(",").map((h) => h.trim()).filter(Boolean)];
  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: true,
      allowedHosts,
      proxy: {
        "/api": { target, changeOrigin: false },
      },
    },
    build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1600 },
  };
});
