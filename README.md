# 🎵 Audio Separator

Веб-приложение для разделения аудио на 6 дорожек прямо в браузере. Полностью клиентское — файлы никуда не загружаются.

## ✨ Возможности

- 🔒 **100% приватность** — вся обработка происходит локально в браузере
- 🧠 **UVR-MDX-NET** — быстрая и надёжная модель разделения вокала
- 🎵 **2 дорожки** — Vocals и Instrumental
- 🌐 **WebGPU + WASM** — использует WebGPU при наличии, fallback на WASM
- 💾 **Кэширование** — модели кэшируются в CacheStorage после первой загрузки
- ⬇️ **Экспорт** — скачивание результатов в WAV формате

## 🧠 Модели

Используются **UVR-MDX-NET** — проверенные модели для разделения вокала:

| Модель | Размер | Описание |
|--------|--------|----------|
| UVR-MDX-NET 9482 | 28 MB | Быстрая, хорошее качество |
| UVR-MDX-NET Voc_FT | 64 MB | Высокое качество |

### Выходы (2 stems):
1. 🎤 **Vocals** — вокал
2. 🎸 **Instrumental** — всё остальное

### Источник моделей:
- **HuggingFace**: [Blane187/all_public_uvr_models](https://huggingface.co/Blane187/all_public_uvr_models)
- **Лицензия**: MIT
- **Архитектура**: MDX-Net от kuielab
- **Оптимизированы** для работы в браузере через onnxruntime-web

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

- **Модель:** UVR-MDX-NET (MDX-Net)
- **Вход:** Стерео аудио (MP3, WAV, OGG, FLAC, M4A, AAC, WebM)
- **Выход:** 2 WAV файла (vocals, instrumental)
- **STFT:** n_fft=768-1024, hop_length=384-512, hann window, center=True
- **Чанк:** 262144 samples (~6s @ 44.1kHz)
- **Overlap:** 50% overlap-add для плавности

## 🔄 Как это работает

1. **Загрузка аудио** — файл декодируется через Web Audio API
2. **Загрузка модели** — ONNX модель скачивается с HuggingFace и кэшируется
3. **Разбиение на чанки** — аудио разбивается на ~6-секундные сегменты с 50% перекрытием
4. **STFT** — каждый чанк конвертируется в magnitude spectrogram
5. **Инференс** — spectrogram подаётся в модель, которая возвращает маску вокала
6. **Применение маски** — маска применяется к stereo STFT
7. **iSTFT** — маскированные спектрограммы конвертируются обратно в аудио
8. **Overlap-add** — чанки собираются обратно с учётом перекрытий
9. **Instrumental** — вычисляется как original - vocals
10. **Экспорт** — результаты кодируются в WAV для воспроизведения/скачивания

## ⚡ Производительность

- **WebGPU:** ~0.3-0.7x realtime (зависит от GPU)
- **WASM:** ~1-3x realtime (зависит от CPU)
- **Первая загрузка:** 28-64 MB (далее из кэша)

## 🐛 Известные ограничения

- Первая загрузка модели занимает время (336 MB)
- Разделение на слабых машинах может занять несколько минут
- GitHub Pages лимит трафика: 100 GB/месяц
- WebGPU может не работать на старых GPU или в некоторых браузерах
- Если HuggingFace заблокирован — нужен VPN для первой загрузки

## 📄 Лицензия

MIT

## 🙏 Благодарности

- **Модели:** [Blane187/all_public_uvr_models](https://huggingface.co/Blane187/all_public_uvr_models)
- **Архитектура:** [kuielab/MDX-Net](https://github.com/kuielab) (Music Demixing Challenge)
- **UVR Project:** [Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- **ONNX Runtime:** [onnxruntime-web](https://github.com/microsoft/onnxruntime)
