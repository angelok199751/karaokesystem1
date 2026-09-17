# 🎵 Audio Separator

Веб-приложение для разделения аудио на вокал и инструментал прямо в браузере. Полностью клиентское — файлы никуда не загружаются.

## ✨ Возможности

- 🔒 **100% приватность** — вся обработка происходит локально в браузере
- 🧠 **ML-модели** — UVR-MDX-NET (проверенные модели разделения вокала)
- 🌐 **WebGPU + WASM** — использует WebGPU при наличии, fallback на WASM
- 💾 **Кэширование** — модели кэшируются в CacheStorage после первой загрузки
- ⬇️ **Экспорт** — скачивание результатов в WAV формате
- 🇷🇺 **Работает в РФ** — модели загружаются с GitHub (не заблокирован)

## 🧠 Модели

Используются модели **UVR-MDX-NET** из проекта [Ultimate Vocal Remover](https://github.com/TRvlvr/model_repo):

| Модель | Размер | Описание |
|--------|--------|----------|
| UVR-MDX-NET 9482 | 28 MB | Быстрая, хорошее качество |
| UVR-MDX-NET Voc_FT | 64 MB | Высокое качество |
| UVR-MDX-NET Inst_HQ_4 | 56 MB | Альтернативная модель |

### 🌐 Система загрузки

Модели загружаются с **GitHub Releases** через систему прокси:

1. **mirror.ghproxy.com** - основной прокси
2. **gh-proxy.com** - альтернативный прокси
3. **ghfast.top** - дополнительный прокси
4. **Прямой доступ** - если прокси недоступны

Приложение автоматически пробует все источники по очереди. Если один не работает, переключается на следующий.

### ⚠️ Если загрузка не работает

Если все прокси не работают в вашем регионе:
- Используйте VPN
- Скачайте модель вручную с [GitHub Releases](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)
- Поместите файл в кэш браузера через DevTools → Application → Cache Storage

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
3. **Размер модели:** Первый визит загружает 28-64 MB модели (далее кэшируется)
4. **CORS:** Модели загружаются с GitHub Releases (CORS включён)

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
2. **Загрузка модели** — ONNX модель скачивается с GitHub Releases и кэшируется
3. **STFT** — аудио конвертируется в спектрограмму (Short-Time Fourier Transform)
4. **Инференс** — спектрограмма пропускается через нейросеть
5. **Маскирование** — выход модели используется как маска для спектрограммы
6. **iSTFT** — маскированные спектрограммы конвертируются обратно в аудио
7. **Экспорт** — результаты кодируются в WAV для воспроизведения/скачивания

## 📊 Технические детали

- **Sample Rate:** 44100 Hz
- **N_FFT:** 768 или 1024 (зависит от модели)
- **Hop Length:** 384 или 512
- **Chunk Size:** 262144 samples (~6 секунд)
- **Overlap:** 2x (50% перекрытие)
- **Формат входа:** `[1, 1, n_freq, n_frames]` — magnitude spectrogram
- **Формат выхода:** `[1, 1, n_freq, n_frames]` — mask

## ⚡ Производительность

- **WebGPU:** ~0.5-1x realtime (зависит от GPU)
- **WASM:** ~2-5x realtime (зависит от CPU)
- **Первая загрузка:** 28-64 MB (далее из кэша)

## 🐛 Известные ограничения

- Первая загрузка модели занимает время (28-64 MB)
- Разделение на слабых машинах может занять несколько минут
- GitHub Pages лимит трафика: 100 GB/месяц
- WebGPU может не работать на старых GPU или в некоторых браузерах
- Качество разделения зависит от выбранной модели

## 📄 Лицензия

MIT

## 🙏 Благодарности

- **Модели:** [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- **ONNX экспорт:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **ONNX Runtime:** [onnxruntime-web](https://github.com/microsoft/onnxruntime)
