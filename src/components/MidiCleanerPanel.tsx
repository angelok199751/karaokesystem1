import { useState } from 'react';
import { CleanerOptions } from '../utils/transcription/midiCleaner';

interface MidiCleanerPanelProps {
  onClean: (options: CleanerOptions) => void;
}

export function MidiCleanerPanel({ onClean }: MidiCleanerPanelProps) {
  const [options, setOptions] = useState<CleanerOptions>({
    minNoteDuration: 0.08,
    maxNoteDuration: 5.0,
    mergeLegato: true,
    legatoThreshold: 0.05,
    minGapBetweenNotes: 0.02,
  });

  const handleClean = () => {
    onClean(options);
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <h3 className="text-lg font-semibold">🧹 MIDI Cleaner</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Мин. длительность ноты (сек)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            value={options.minNoteDuration}
            onChange={(e) => setOptions({ ...options, minNoteDuration: Number(e.target.value) })}
            className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
          />
          <div className="text-xs text-gray-500 mt-1">
            Удаляет ноты короче этого значения
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Макс. длительность ноты (сек)
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            value={options.maxNoteDuration}
            onChange={(e) => setOptions({ ...options, maxNoteDuration: Number(e.target.value) })}
            className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
          />
          <div className="text-xs text-gray-500 mt-1">
            Удаляет ноты длиннее этого значения
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Порог легато (сек)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            value={options.legatoThreshold}
            onChange={(e) => setOptions({ ...options, legatoThreshold: Number(e.target.value) })}
            className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
          />
          <div className="text-xs text-gray-500 mt-1">
            Сливает соседние ноты одной высоты
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Мин. промежуток (сек)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            value={options.minGapBetweenNotes}
            onChange={(e) => setOptions({ ...options, minGapBetweenNotes: Number(e.target.value) })}
            className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
          />
          <div className="text-xs text-gray-500 mt-1">
            Убирает перекрывающиеся ноты
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="mergeLegato"
          checked={options.mergeLegato}
          onChange={(e) => setOptions({ ...options, mergeLegato: e.target.checked })}
          className="w-4 h-4"
        />
        <label htmlFor="mergeLegato" className="text-sm text-gray-300">
          Слияние легато (соседние ноты одной высоты)
        </label>
      </div>

      <button
        onClick={handleClean}
        className="w-full px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded-lg font-medium transition"
      >
        🧹 Очистить MIDI
      </button>
    </div>
  );
}
