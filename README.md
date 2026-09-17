# 🎵 Audio Separator

Веб-приложение для разделения аудио на 6 дорожек прямо в браузере. Полностью клиентское — файлы никуда не загружаются.

## ✨ Возможности

- 🔒 **100% приватность** — вся обработка происходит локально в браузере
- 🧠 **HT-Demucs FT** — state-of-the-art модель разделения (SDR 9.19 dB для вокала)
- 🎼 **Вокал → MIDI** — транскрипция вокала в MIDI с помощью Spotify Basic Pitch
- 🎵 **2 дорожки** — Vocals и Instrumental
- 🌐 **WebGPU + WASM** — использует WebGPU при наличии, fallback на WASM
- 💾 **Кэширование** — модели кэшируются в CacheStorage после первой загрузки
- ⬇️ **Экспорт** — скачивание результатов в WAV и MIDI форматах

## 🧠 Модель

Используется **Demucs HT** — оптимизированная версия для браузера через пакет `demucs-web`:

| Модель | Размер | Описание |
|--------|--------|----------|
| Demucs HT Embedded | 170 MB | Оптимизирована для браузера |

### Выходы (2 stems):
1. 🎤 **Vocals** — вокал
2. 🎸 **Instrumental** — drums + bass + other

### Источник модели:
- **HuggingFace**: [timcsy/demucs-web-onnx](https://huggingface.co/timcsy/demucs-web-onnx)
- **Пакет**: [demucs-web](https://www.npmjs.com/package/demucs-web)
- **Лицензия**: MIT
- **Архитектура**: Hybrid Transformer Demucs от Facebook Research
- **Оптимизация**: Специально для браузерного использования с WebGPU/WASM

## 🎼 Транскрипция вокала в MIDI

После разделения аудио можно конвертировать вокальную дорожку в MIDI файл:

### Модель
Используется **Spotify Basic Pitch** — лёгкая модель для транскрипции аудио в MIDI:

| Параметр | Значение |
|----------|----------|
| Размер | ~230 KB |
| Вход | Аудио 22050 Hz, моно, окно 2 секунды |
| Выход | MIDI ноты с velocity |
| Лицензия | Apache-2.0 |

### Как это работает
1. **Подготовка аудио** — вокал ресемплится в 22050 Hz и конвертируется в моно
2. **Нарезка на окна** — аудио разбивается на перекрывающиеся окна по 2 секунды
3. **Инференс** — каждое окно обрабатывается моделью Basic Pitch
4. **Постобработка** — тензоры конвертируются в MIDI ноты (start, end, pitch, velocity)
5. **Генерация MIDI** — создаётся Standard MIDI File (format 0)

### Использование
1. Разделите аудио на вокал и инструментал
2. Нажмите кнопку "🎼 Создать MIDI" рядом с вокальной дорожкой
3. Дождитесь завершения транскрипции (10-30 секунд для 3-минутного трека)
4. Скачайте MIDI файл кнопкой "⬇️ Download MIDI"

### Ограничения
- Транскрипция не идеальна для сложного вокала с сильным вибрато или эффектами
- MIDI может потребовать ручной чистки в DAW
- Модель покрывает диапазон MIDI 21-108 (пиано)

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

- **Модель:** Demucs HT (через demucs-web)
- **Вход:** Стерео аудио (MP3, WAV, OGG, FLAC, M4A, AAC, WebM)
- **Выход:** 2 WAV файла (vocals, instrumental)
- **Формат входа:** waveform [1, 2, 343980] + magSpec [1, 4, 2048, 336]
- **Формат выхода:** 4 дорожки (drums, bass, other, vocals)
- **Чанк:** 343980 samples (7.8s @ 44.1kHz)
- **Overlap:** 25% overlap-add для плавности

## 🔄 Как это работает

1. **Загрузка аудио** — файл декодируется через Web Audio API
2. **Загрузка модели** — demucs-web загружает оптимизированную модель (170MB) и кэширует её
3. **Разбиение на чанки** — аудио разбивается на 7.8-секундные сегменты с 25% перекрытием
4. **Подготовка входа** — для каждого чанка создаётся waveform и magnitude spectrogram
5. **Инференс** — модель обрабатывает оба входа и возвращает 4 дорожки
6. **Извлечение stems** — Vocals = vocals stem, Instrumental = drums + bass + other
7. **Overlap-add** — чанки собираются обратно с учётом перекрытий
8. **Нормализация** — сигнал нормализуется до 0.95 peak
9. **Экспорт** — результаты кодируются в WAV для воспроизведения/скачивания

## ⚡ Производительность

- **WebGPU:** ~0.5x realtime (зависит от GPU)
- **WASM:** ~2-3x realtime (зависит от CPU)
- **Первая загрузка:** 170 MB (далее из кэша)
- **3-минутная песня:** ~60-90s (зависит от оборудования)

## 🐛 Известные ограничения

- Первая загрузка модели занимает время (170 MB)
- Разделение на слабых машинах может занять несколько минут
- GitHub Pages лимит трафика: 100 GB/месяц
- WebGPU может не работать на старых GPU или в некоторых браузерах
- **VPN может не понадобиться** — приложение автоматически пробует несколько зеркал (hf-mirror.com, hub.nuaa.cf)
- Если все зеркала недоступны — нужен VPN для первой загрузки
- После первой загрузки модель кэшируется в браузере
- Модель оптимизирована для браузера, но всё ещё требует значительных ресурсов

## 📄 Лицензия

MIT

## 🙏 Благодарности

- **Пакет:** [demucs-web](https://www.npmjs.com/package/demucs-web) — готовая реализация для браузера
- **Модель:** [timcsy/demucs-web-onnx](https://huggingface.co/timcsy/demucs-web-onnx) — оптимизированная ONNX версия
- **Архитектура:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs) (Hybrid Transformer Demucs)
- **ONNX Runtime:** [onnxruntime-web](https://github.com/microsoft/onnxruntime)
