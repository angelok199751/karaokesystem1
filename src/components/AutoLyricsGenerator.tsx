import { useState } from 'react';
import { KaraokeJSON } from '../utils/transcription/karaokeJSON';
import { generateAutoLyrics, downloadAutoLyrics, AutoLyricsOptions } from '../utils/transcription/autoLyricsGenerator';
import { alignNotesToLyrics, downloadAlignedKaraoke } from '../utils/transcription/lyricsAligner';

interface AutoLyricsGeneratorProps {
  karaokeJSON: KaraokeJSON;
  onLyricsGenerated: (lyrics: any) => void;
}

export function AutoLyricsGenerator({ karaokeJSON, onLyricsGenerated }: AutoLyricsGeneratorProps) {
  const [options, setOptions] = useState<AutoLyricsOptions>({
    placeholderText: 'la',
    phraseGap: 0.5,
    useSyllables: false,
  });

  const [generatedLyrics, setGeneratedLyrics] = useState<any>(null);

  const handleGenerate = () => {
    const lyrics = generateAutoLyrics(karaokeJSON, options);
    setGeneratedLyrics(lyrics);
    onLyricsGenerated(lyrics);
  };

  const handleDownloadLyrics = () => {
    if (generatedLyrics) {
      downloadAutoLyrics(generatedLyrics, 'lyrics_auto.json');
    }
  };

  const handleDownloadAligned = () => {
    if (generatedLyrics) {
      const aligned = alignNotesToLyrics(karaokeJSON, generatedLyrics);
      downloadAlignedKaraoke(aligned, 'karaoke_aligned.json');
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <h3 className="text-lg font-semibold">🎤 Автоматическая генерация текста</h3>
      
      <div className="space-y-3">
        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Текст-заглушка
          </label>
          <input
            type="text"
            value={options.placeholderText}
            onChange={(e) => setOptions({ ...options, placeholderText: e.target.value })}
            className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
            placeholder="la"
          />
          <div className="text-xs text-gray-500 mt-1">
            Текст для каждой ноты (например: "la", "♪", "да")
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="useSyllables"
            checked={options.useSyllables}
            onChange={(e) => setOptions({ ...options, useSyllables: e.target.checked })}
            className="w-4 h-4"
          />
          <label htmlFor="useSyllables" className="text-sm text-gray-300">
            Каждая нота = отдельный слог
          </label>
        </div>

        <div className="text-xs text-gray-500">
          {options.useSyllables 
            ? 'Каждая нота будет отдельным сегментом с текстом-заглушкой'
            : 'Каждая фраза будет одним сегментом с повторяющимся текстом'
          }
        </div>
      </div>

      <button
        onClick={handleGenerate}
        className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition"
      >
        🎤 Сгенерировать текст
      </button>

      {generatedLyrics && (
        <div className="space-y-2">
          <div className="bg-green-500/20 border border-green-500/30 rounded p-3 text-sm text-green-300">
            <strong>✓ Сгенерировано:</strong> {generatedLyrics.lyrics.length} сегментов
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleDownloadLyrics}
              className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm transition"
            >
              ⬇️ lyrics.json
            </button>
            <button
              onClick={handleDownloadAligned}
              className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-sm transition"
            >
              ⬇️ Выровненный
            </button>
          </div>

          <div className="text-xs text-gray-500">
            💡 Отредактируйте текст в JSON редакторе или скачайте готовый файл
          </div>
        </div>
      )}
    </div>
  );
}
