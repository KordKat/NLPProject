import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  const target = env.VITE_API_PROXY || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: true,
      proxy: {
        "/api": { target, changeOrigin: false },
      },
    },
    build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1600 },
  };
});
