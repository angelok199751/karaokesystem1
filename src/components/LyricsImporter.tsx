import { useState, useRef } from 'react';
import { LyricsFile } from '../utils/transcription/lyricsAligner';

interface LyricsImporterProps {
  onImport: (lyrics: LyricsFile) => void;
}

export function LyricsImporter({ onImport }: LyricsImporterProps) {
  const [error, setError] = useState<string | null>(null);
  const [lyrics, setLyrics] = useState<LyricsFile | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);

    try {
      const text = await file.text();
      const parsed = JSON.parse(text);

      // Валидация формата
      if (!parsed.title || !Array.isArray(parsed.lyrics)) {
        throw new Error('Invalid lyrics file format. Expected {title: string, lyrics: Array<{start, end, text}>}');
      }

      // Валидация каждого сегмента
      for (const segment of parsed.lyrics) {
        if (typeof segment.start !== 'number' || 
            typeof segment.end !== 'number' || 
            typeof segment.text !== 'string') {
          throw new Error('Invalid segment format. Each segment must have {start: number, end: number, text: string}');
        }
      }

      setLyrics(parsed);
      onImport(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse lyrics file');
      setLyrics(null);
    }
  };

  const handleClear = () => {
    setLyrics(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">📝 Импорт текста</h3>
        {lyrics && (
          <button
            onClick={handleClear}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition"
          >
            ✕ Очистить
          </button>
        )}
      </div>

      <div className="space-y-2">
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleFileChange}
          className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 file:mr-4 file:py-1 file:px-4 file:rounded file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700"
        />
        
        <div className="text-xs text-gray-400">
          Формат: JSON с полями {`{title: string, lyrics: Array<{start, end, text}>}`}
        </div>
      </div>

      {error && (
        <div className="bg-red-500/20 border border-red-500/30 rounded p-3 text-sm text-red-300">
          <strong>Ошибка:</strong> {error}
        </div>
      )}

      {lyrics && (
        <div className="bg-green-500/20 border border-green-500/30 rounded p-3 text-sm text-green-300">
          <strong>✓ Загружено:</strong> {lyrics.title} ({lyrics.lyrics.length} сегментов)
        </div>
      )}
    </div>
  );
}
