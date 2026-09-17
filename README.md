# 🎵 Audio Separator

Веб-приложение для разделения аудио на 6 дорожек прямо в браузере. Полностью клиентское — файлы никуда не загружаются.

## ✨ Возможности

- 🔒 **100% приватность** — вся обработка происходит локально в браузере
- 🧠 **HT-Demucs FT** — state-of-the-art модель разделения (SDR 9.19 dB для вокала)
- 🎵 **2 дорожки** — Vocals и Instrumental
- 🌐 **WebGPU + WASM** — использует WebGPU при наличии, fallback на WASM
- 💾 **Кэширование** — модели кэшируются в CacheStorage после первой загрузки
- ⬇️ **Экспорт** — скачивание результатов в WAV формате

## 🧠 Модель

Используется **HT-Demucs FT** — первая полная ONNX-версия с проверенным паритетом:

| Модель | Размер | SDR (вокал) | Описание |
|--------|--------|-------------|----------|
| HT-Demucs FT Vocals | 316 MB | 9.19 dB | Лучший open-source результат |

### Выходы (2 stems):
1. 🎤 **Vocals** — вокал (SDR 9.19 dB)
2. 🎸 **Instrumental** — drums + bass + other

### Источник модели:
- **HuggingFace**: [StemSplitio/htdemucs-ft-vocals-onnx](https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx)
- **Лицензия**: MIT
- **Архитектура**: Hybrid Transformer Demucs от Facebook Research
- **Паритет**: Проверен с PyTorch версией (< 1e-3 max abs diff)

## 🌐 Загрузка модели

Модели загружаются с HuggingFace (28-64 MB). Если HuggingFace заблокирован в вашем регионе:
- Используйте VPN для первой загрузки
- После первой загрузки модель кэшируется в браузере
- Повторные запуски используют кэш (мгновенно)
- Есть fallback на hf-mirror.com

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

## 📊 Технические детали

- **Модель:** HT-Demucs FT (Hybrid Transformer Demucs)
- **Вход:** Стерео аудио (MP3, WAV, OGG, FLAC, M4A, AAC, WebM)
- **Выход:** 2 WAV файла (vocals, instrumental)
- **Формат входа:** (1, 2, 343980) - стерео аудио напрямую
- **Формат выхода:** (1, 4, 2, 343980) - 4 дорожки (drums, bass, other, vocals)
- **Чанк:** 343980 samples (7.8s @ 44.1kHz)
- **Overlap:** 25% overlap-add для плавности

## 🔄 Как это работает

1. **Загрузка аудио** — файл декодируется через Web Audio API
2. **Загрузка модели** — ONNX модель скачивается с HuggingFace и кэшируется
3. **Разбиение на чанки** — аудио разбивается на 7.8-секундные сегменты с 25% перекрытием
4. **Инференс** — каждый чанк подаётся в модель напрямую (без STFT!)
5. **Извлечение stems** — модель возвращает 4 дорожки: drums, bass, other, vocals
6. **Объединение** — Vocals = vocals stem, Instrumental = drums + bass + other
7. **Overlap-add** — чанки собираются обратно с учётом перекрытий
8. **Нормализация** — сигнал нормализуется до 0.95 peak
9. **Экспорт** — результаты кодируются в WAV для воспроизведения/скачивания

## ⚡ Производительность

- **WebGPU:** ~0.5x realtime (зависит от GPU)
- **WASM:** ~2-3x realtime (зависит от CPU)
- **Первая загрузка:** 316 MB (далее из кэша)
- **3-минутная песня:** ~88s на M4 Pro CPU

## 🐛 Известные ограничения

- Первая загрузка модели занимает время (316 MB)
- Разделение на слабых машинах может занять несколько минут
- GitHub Pages лимит трафика: 100 GB/месяц
- WebGPU может не работать на старых GPU или в некоторых браузерах
- Если HuggingFace заблокирован — нужен VPN для первой загрузки

## 📄 Лицензия

MIT

## 🙏 Благодарности

- **Модель:** [StemSplitio/htdemucs-ft-onnx](https://huggingface.co/StemSplitio/htdemucs-ft-onnx)
- **Архитектура:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs) (Hybrid Transformer Demucs)
- **ONNX экспорт:** [StemSplit](https://stemsplit.io/)
- **ONNX Runtime:** [onnxruntime-web](https://github.com/microsoft/onnxruntime)
