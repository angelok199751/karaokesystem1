# 🎵 Audio Separator

Веб-приложение для разделения аудио на 6 дорожек прямо в браузере. Полностью клиентское — файлы никуда не загружаются.

## ✨ Возможности

- 🔒 **100% приватность** — вся обработка происходит локально в браузере
- 🧠 **BS-Roformer-SW** — state-of-the-art модель разделения на 6 stems
- 🎵 **6 дорожек** — Bass, Drums, Other, Vocals, Guitar, Piano
- 🌐 **WebGPU + WASM** — использует WebGPU при наличии, fallback на WASM
- 💾 **Кэширование** — модели кэшируются в CacheStorage после первой загрузки
- ⬇️ **Экспорт** — скачивание результатов в WAV формате

## 🧠 Модель

Используется **BS-Roformer-SW** — state-of-the-art модель разделения аудио:

| Модель | Размер | Описание |
|--------|--------|----------|
| BS-Roformer-SW FP16 | 336 MB | Рекомендуемая, лучшее качество (Vocals + Instrumental) |
| BS-Roformer-SW FP32 | 669 MB | Fallback, если FP16 не работает (Vocals + Instrumental) |

### Выходы модели (2 stems):
1. 🎤 **Vocals** — вокал
2. 🎸 **Instrumental** — всё остальное (бас, ударные, гитара, пианино и другие инструменты)

### Источник модели:
- **HuggingFace**: [elicwhite/bs-roformer-sw-6stem-onnx](https://huggingface.co/elicwhite/bs-roformer-sw-6stem-onnx)
- **Лицензия**: MIT
- **Архитектура**: Band-Split RoFormer от ByteDance AI Labs
- **Специально подготовлена** для работы в браузере через onnxruntime-web

## 🌐 Загрузка модели

Модель загружается с HuggingFace. Если HuggingFace заблокирован в вашем регионе:
- Используйте VPN для первой загрузки
- После первой загрузки модель кэшируется в браузере
- Повторные запуски используют кэш (мгновенно)

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

- **Модель:** BS-Roformer-SW (Band-Split RoFormer)
- **Вход:** Стерео аудио (MP3, WAV, OGG, FLAC, M4A, AAC, WebM)
- **Выход:** 2 WAV файла (vocals, instrumental)
- **STFT:** n_fft=2048, hop_length=512, hann window, center=True
- **Чанк:** 176400 samples (4s @ 44.1kHz, T=345 frames)
- **Overlap:** 25% overlap-add для плавности

## 🔄 Как это работает

1. **Загрузка аудио** — файл декодируется через Web Audio API
2. **Загрузка модели** — ONNX модель скачивается с HuggingFace и кэшируется
3. **Разбиение на чанки** — аудио разбивается на 4-секундные сегменты с 25% перекрытием
4. **STFT** — каждый чанк конвертируется в спектрограмму (комплексные числа)
5. **Инференс** — спектрограммы подаются в модель, которая возвращает маски для всех stems
6. **iSTFT** — маскированные спектрограммы конвертируются обратно в аудио
7. **Объединение** — вокал (stem 3) и инструментал (все остальные stems объединены)
8. **Overlap-add** — чанки собираются обратно с учётом перекрытий
9. **Экспорт** — результаты кодируются в WAV для воспроизведения/скачивания

## ⚡ Производительность

- **WebGPU:** ~0.5-1x realtime (зависит от GPU)
- **WASM:** ~2-5x realtime (зависит от CPU)
- **Первая загрузка:** 336 MB (далее из кэша)

## 🐛 Известные ограничения

- Первая загрузка модели занимает время (336 MB)
- Разделение на слабых машинах может занять несколько минут
- GitHub Pages лимит трафика: 100 GB/месяц
- WebGPU может не работать на старых GPU или в некоторых браузерах
- Если HuggingFace заблокирован — нужен VPN для первой загрузки

## 📄 Лицензия

MIT

## 🙏 Благодарности

- **Модель:** [elicwhite/bs-roformer-sw-6stem-onnx](https://huggingface.co/elicwhite/bs-roformer-sw-6stem-onnx)
- **Архитектура:** [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
- **Обучение:** [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- **ONNX Runtime:** [onnxruntime-web](https://github.com/microsoft/onnxruntime)
- **Оригинальная статья:** [Band-Split RoFormer](https://arxiv.org/abs/2309.02612) от ByteDance AI Labs
