import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { rmSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';

function removeWasmPlugin() {
  return {
    name: 'remove-wasm',
    closeBundle() {
      const assetsDir = join(process.cwd(), 'dist/assets');
      if (existsSync(assetsDir)) {
        const files = readdirSync(assetsDir);
        files.filter((f) => f.endsWith('.wasm')).forEach((f) => {
          rmSync(join(assetsDir, f));
          console.log(`Removed ${f} from dist (loaded from CDN)`);
        });
      }
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), removeWasmPlugin()],
  server: {
    host: "0.0.0.0",
    port: 3000,
    strictPort: true,
    hmr: {
      port: 3000,
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'ort': ['onnxruntime-web'],
        },
      },
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
