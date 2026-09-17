import { useState } from 'react';
import { MidiEditor } from './MidiEditor';
import { JsonEditor } from './JsonEditor';
import { KaraokeJSON, exportKaraokeJSON, downloadKaraokeJSON } from '../utils/transcription/karaokeJSON';

interface MidiEditorModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialData: KaraokeJSON;
  filename: string;
}

export function MidiEditorModal({ isOpen, onClose, initialData, filename }: MidiEditorModalProps) {
  const [karaokeJSON, setKaraokeJSON] = useState<KaraokeJSON>(initialData);
  const [activeTab, setActiveTab] = useState<'visual' | 'json'>('visual');

  if (!isOpen) return null;

  const handleSave = () => {
    downloadKaraokeJSON(karaokeJSON, filename);
  };

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-xl max-w-7xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div>
            <h2 className="text-2xl font-bold">🎼 MIDI Редактор</h2>
            <p className="text-sm text-gray-400">Редактируйте ноты и тайминги</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition"
            >
              💾 Сохранить JSON
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition"
            >
              ✕ Закрыть
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700">
          <button
            onClick={() => setActiveTab('visual')}
            className={`px-6 py-3 font-medium transition ${
              activeTab === 'visual'
                ? 'bg-gray-800 text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            🎹 Визуальный редактор
          </button>
          <button
            onClick={() => setActiveTab('json')}
            className={`px-6 py-3 font-medium transition ${
              activeTab === 'json'
                ? 'bg-gray-800 text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            📝 JSON редактор
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {activeTab === 'visual' ? (
            <MidiEditor karaokeJSON={karaokeJSON} onUpdate={setKaraokeJSON} />
          ) : (
            <JsonEditor value={karaokeJSON} onChange={setKaraokeJSON} />
          )}
        </div>
      </div>
    </div>
  );
}
