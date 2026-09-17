import { ModelConfig } from '../utils/modelManager';

interface ModelSelectorProps {
  models: ModelConfig[];
  selected: ModelConfig;
  onChange: (model: ModelConfig) => void;
  disabled: boolean;
}

export function ModelSelector({ models, selected, onChange, disabled }: ModelSelectorProps) {
  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-gray-300 mb-2">
        🧠 Separation Model
      </label>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {models.map((model) => (
          <button
            key={model.id}
            onClick={() => onChange(model)}
            disabled={disabled}
            className={`
              p-4 rounded-xl border text-left transition-all
              ${selected.id === model.id
                ? 'border-violet-500 bg-violet-500/10 ring-1 ring-violet-500/50'
                : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className={`w-3 h-3 rounded-full ${
                selected.id === model.id ? 'bg-violet-400' : 'bg-gray-600'
              }`} />
              <span className="font-medium text-white text-sm">{model.name}</span>
            </div>
            <p className="text-xs text-gray-400 line-clamp-2">{model.description}</p>
            <div className="flex items-center gap-2 mt-2">
              <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-gray-300">
                {model.size}
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-gray-300">
                {model.stems.length} stems
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-gray-300">
                MIT License
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
