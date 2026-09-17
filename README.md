# 🎵 Audio Separator

Веб-приложение для разделения аудио на вокал и инструментал прямо в браузере. Полностью клиентское — файлы никуда не загружаются.

## ✨ Возможности

- 🔒 **100% приватность** — вся обработка происходит локально в браузере
- 🧠 **ML-модель** — BS PolarFormer (BSRoformer с PoPE embeddings)
- 🌐 **WebGPU + WASM** — использует WebGPU при наличии, fallback на WASM
- 💾 **Кэширование** — модели кэшируются в CacheStorage после первой загрузки
- ⬇️ **Экспорт** — скачивание результатов в WAV формате
- 🎚️ **Стерео** — сохраняет стерео при разделении

## 🧠 Модель

Используется **BS PolarFormer** — современная модель разделения источников:
- Архитектура: BSRoformer с PoPE (Polar Positional Embeddings)
- Параметры: 51M
- Качество: SDR 11.0 на вокале (Multisong Dataset)
- Форматы: FP16 (103 MB) и FP32 (201 MB)
- Источник: [bgkb/bs_polarformer](https://huggingface.co/bgkb/bs_polarformer)

## 🌐 Поддержка браузеров

| Функция | Chrome | Edge | Safari | Firefox |
|---------|--------|------|--------|---------|
| WebGPU | 113+ | 113+ | 18+ (TP) | 121+ (exp) |
| WASM | ✅ | ✅ | ✅ | ✅ |

**Примечание:** WebGPU обеспечивает в 3-5 раз более быстрый инференс. WASM работает везде, но медленнее.

## 🚀 Локальный запуск

```bash
npm install
npm run dev
```

## 📦 Сборка

```bash
npm run build
```

Результат в папке `dist/` — можно деплоить на любой статический хостинг.

## 🌍 Деплой на GitHub Pages

### Через GitHub Actions (рекомендуется)

Создай `.github/workflows/deploy.yml`:

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

Затем: **Settings → Pages → Source → GitHub Actions**

## ⚠️ Важные нюансы для GitHub Pages

1. **Base Path:** В Vite конфигурации используется `base: './'` для относительных путей
2. **Нет SharedArrayBuffer:** GitHub Pages не может устанавливать COOP/COEP заголовки, поэтому используется однопоточный WASM
3. **Размер модели:** Первый визит загружает 103-201 MB модели (далее кэшируется)
4. **CORS:** Модели загружаются с Hugging Face (CORS включён)

## 🏗️ Архитектура

```
src/
├── App.tsx                    # Главный компонент
├── components/
│   ├── AudioPlayer.tsx        # Аудиоплеер с seek
│   ├── FileUpload.tsx         # Drag & drop загрузка
│   ├── ModelSelector.tsx      # Выбор модели
│   └── ProgressBar.tsx        # Индикаторы прогресса
└── utils/
    ├── modelManager.ts        # Загрузка моделей, кэш, ONNX сессии
    └── separation.ts          # Логика разделения (STFT, инференс, iSTFT)
```

## 🔄 Как это работает

1. **Загрузка аудио** — файл декодируется через Web Audio API
2. **Загрузка модели** — ONNX модель скачивается с Hugging Face и кэшируется
3. **STFT** — аудио конвертируется в спектрограмму (Short-Time Fourier Transform)
4. **Инференс** — спектрограмма пропускается через нейросеть
5. **Маскирование** — выход модели используется как комплексная маска
6. **iSTFT** — маскированные спектрограммы конвертируются обратно в аудио
7. **Экспорт** — результаты кодируются в WAV для воспроизведения/скачивания

## 📊 Технические детали

- **Sample Rate:** 44100 Hz
- **N_FFT:** 2048
- **Hop Length:** 512
- **Chunk Size:** 131072 samples (~3 секунды)
- **Overlap:** 2x (50% перекрытие)
- **Формат входа:** `[batch, time_frames, 4100]` — interleaved stereo STFT
- **Формат выхода:** `[1, 1, 2050, time_frames, 2]` — комплексная маска

## ⚡ Производительность

- **WebGPU:** ~0.5-1x realtime (зависит от GPU)
- **WASM:** ~2-5x realtime (зависит от CPU)
- **Первая загрузка:** 103-201 MB (далее из кэша)

## 🐛 Известные ограничения

- Первая загрузка модели занимает время (103-201 MB)
- Разделение на слабых машинах может занять несколько минут
- GitHub Pages лимит трафика: 100 GB/месяц
- WebGPU может не работать на старых GPU или в некоторых браузерах

## 📄 Лицензия

MIT

## 🙏 Благодарности

- **Модель:** [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- **Архитектура:** [lucidrains](https://github.com/lucidrains) BSRoformer
- **ONNX Runtime:** [onnxruntime-web](https://github.com/microsoft/onnxruntime)
- **Реализация:** Основано на [bgkb/bs_polarformer](https://huggingface.co/bgkb/bs_polarformer)
