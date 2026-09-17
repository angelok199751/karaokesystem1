import { useState } from 'react';
import { VocalFilterOptions, DEFAULT_VOCAL_FILTER_OPTIONS } from '../utils/transcription/vocalFilter';

interface VocalFilterPanelProps {
  options: VocalFilterOptions;
  onChange: (options: VocalFilterOptions) => void;
}

export function VocalFilterPanel({ options, onChange }: VocalFilterPanelProps) {
  const [expanded, setExpanded] = useState(false);

  const handleChange = (key: keyof VocalFilterOptions, value: any) => {
    onChange({ ...options, [key]: value });
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">🎤 Вокальный фильтр</h3>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-sm text-purple-400 hover:text-purple-300"
        >
          {expanded ? '▼ Скрыть' : '▶ Показать'}
        </button>
      </div>

      {expanded && (
        <div className="space-y-4">
          {/* Spectral Filter */}
          <div className="space-y-2">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={options.enableSpectralFilter}
                onChange={(e) => handleChange('enableSpectralFilter', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-gray-300">Спектральная фильтрация</span>
            </label>
            
            {options.enableSpectralFilter && (
              <div className="ml-6 space-y-2">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">
                    Мин. частота: {options.vocalRangeMin} Hz
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="300"
                    value={options.vocalRangeMin}
                    onChange={(e) => handleChange('vocalRangeMin', Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">
                    Макс. частота: {options.vocalRangeMax} Hz
                  </label>
                  <input
                    type="range"
                    min="500"
                    max="2000"
                    value={options.vocalRangeMax}
                    onChange={(e) => handleChange('vocalRangeMax', Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Note Range Filter */}
          <div className="space-y-2">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={options.enableNoteFilter}
                onChange={(e) => handleChange('enableNoteFilter', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-gray-300">Фильтр по диапазону нот</span>
            </label>
            
            {options.enableNoteFilter && (
              <div className="ml-6 space-y-2">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">
                    Мин. нота: MIDI {options.midiRangeMin} ({midiToNoteName(options.midiRangeMin)})
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="60"
                    value={options.midiRangeMin}
                    onChange={(e) => handleChange('midiRangeMin', Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">
                    Макс. нота: MIDI {options.midiRangeMax} ({midiToNoteName(options.midiRangeMax)})
                  </label>
                  <input
                    type="range"
                    min="60"
                    max="100"
                    value={options.midiRangeMax}
                    onChange={(e) => handleChange('midiRangeMax', Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Amplitude Filter */}
          <div className="space-y-2">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={options.enableAmplitudeFilter}
                onChange={(e) => handleChange('enableAmplitudeFilter', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-gray-300">Фильтр по длительности</span>
            </label>
            
            {options.enableAmplitudeFilter && (
              <div className="ml-6">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">
                    Мин. длительность: 50ms
                  </label>
                  <div className="text-xs text-gray-500">
                    Удаляет очень короткие ноты (артефакты)
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Reset button */}
          <button
            onClick={() => onChange(DEFAULT_VOCAL_FILTER_OPTIONS)}
            className="w-full px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition"
          >
            🔄 Сбросить настройки
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Convert MIDI note number to note name
 */
function midiToNoteName(midi: number): string {
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(midi / 12) - 1;
  const note = noteNames[midi % 12];
  return `${note}${octave}`;
}
