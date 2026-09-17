# 🎵 Audio Separator

A fully client-side web application for separating audio into vocals and instrumentals (or multiple stems). All processing happens in the browser — no server, no uploads, complete privacy.

## Features

- 🔒 **100% Private** — All audio processing happens locally in your browser
- 🧠 **ML-powered** — Uses ONNX models (MDX-Net, Kim Vocal 2, HTDemucs) via onnxruntime-web
- 🌐 **WebGPU + WASM** — Leverages WebGPU when available, falls back to WASM
- 💾 **Model Caching** — Models are cached in CacheStorage after first download
- 📦 **Multiple Models** — Choose between fast (2 stems) and quality (4 stems) separation
- ⬇️ **Download Results** — Export separated stems as WAV files

## Tech Stack

- **Frontend:** React + TypeScript + Tailwind CSS
- **Build:** Vite
- **ML Runtime:** onnxruntime-web
- **Audio Processing:** Web Audio API + custom STFT/iSTFT
- **Models:** ONNX format (loaded from Hugging Face)

## Browser Requirements

| Feature | Chrome | Edge | Safari | Firefox |
|---------|--------|------|--------|---------|
| WebGPU | 113+ | 113+ | 18+ (TP) | 121+ (exp) |
| WASM | ✅ | ✅ | ✅ | ✅ |

**Note:** WebGPU provides 3-5x faster inference. WASM works everywhere but is slower.

## Local Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

The build output is in `dist/` and can be deployed to any static hosting.

## Deploy to GitHub Pages

### Option 1: GitHub Actions (Recommended)

1. Create a `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist
      - uses: actions/deploy-pages@v4
```

2. Go to **Settings → Pages → Source → GitHub Actions**

### Option 2: Manual Deploy

```bash
npm run build
# Copy dist/ to gh-pages branch or upload to hosting
```

## Important Notes for GitHub Pages

1. **Base Path:** The app uses `base: './'` in Vite config for relative paths
2. **No SharedArrayBuffer:** GitHub Pages can't set COOP/COEP headers, so we use single-threaded WASM
3. **Model Size:** First visit downloads 30-80MB model (cached afterwards)
4. **CORS:** Models are fetched from Hugging Face (CORS-enabled)

## Architecture

```
src/
├── App.tsx                    # Main application component
├── components/
│   ├── AudioPlayer.tsx        # Custom audio player with seek
│   ├── FileUpload.tsx         # Drag & drop file upload
│   ├── ModelSelector.tsx      # Model selection cards
│   └── ProgressBar.tsx        # Progress indicators
└── utils/
    ├── audioProcessor.ts      # STFT, iSTFT, WAV encoding
    ├── modelManager.ts        # Model download, caching, ONNX session
    └── separation.ts          # Audio separation logic
```

## How It Works

1. **Load Audio** — File is decoded using Web Audio API
2. **Download Model** — ONNX model is fetched from Hugging Face and cached
3. **STFT** — Audio is converted to spectrogram using Short-Time Fourier Transform
4. **Inference** — Spectrogram is passed through the neural network
5. **Masking** — Model output is used as a soft mask on the spectrogram
6. **iSTFT** — Masked spectrograms are converted back to audio
7. **Export** — Results are encoded as WAV for playback/download

## Known Limitations

- First model download takes time (30-80MB depending on model)
- Separation on weak hardware may take several minutes
- Quality depends on the model — MDX-Net is good for vocals, HTDemucs for multi-stem
- GitHub Pages traffic limit: 100GB/month

## License

MIT
