import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// `--mode static` builds the server-less demo (GitHub Pages / static Space): relative asset paths, /api answered in-browser.
export default defineConfig(({ mode }) => ({
  base: mode === "static" ? "./" : "/",
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": "http://localhost:7860" } },
  build: { chunkSizeWarningLimit: 1200 },
}));
