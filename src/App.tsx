import { useState, useRef, useCallback } from 'react';
import { transcribeAudio, downloadMidi, TranscriptionProgress } from './utils/transcription';
import { MODELS, ModelConfig, checkWebGPUAvailability, downloadModel, createSession } from './utils/modelManager';
import { separateAudio, SeparationProgress } from './utils/separation';

type AppState = 'idle' | 'file-loaded' | 'loading-model' | 'processing' | 'done' | 'error';

interface AppProgress {
  modelProgress: number;
  modelMessage: string;
  separationProgress: SeparationProgress | null;
  transcriptionProgress: TranscriptionProgress | null;
}

export default function App() {
  const [state, setState] = useState<AppState>('idle');
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelConfig>(MODELS[0]);
  const [results, setResults] = useState<{ name: string; audioBuffer: AudioBuffer; wavData: ArrayBuffer }[]>([]);
  const [midiData, setMidiData] = useState<Uint8Array | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [webgpuAvailable, setWebgpuAvailable] = useState(false);
  const [progress, setProgress] = useState<AppProgress>({
    modelProgress: 0,
    modelMessage: '',
    separationProgress: null,
    transcriptionProgress: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);

  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    return audioContextRef.current;
  }, []);

  const handleFileSelect = useCallback(async (file: File) => {
    setAudioFile(file);
    setState('file-loaded');
    setResults([]);
    setMidiData(null);
    setErrorMsg('');
    
    try {
      const ctx = getAudioContext();
      const arrayBuffer = await file.arrayBuffer();
      const buffer = await ctx.decodeAudioData(arrayBuffer);
      setAudioBuffer(buffer);
    } catch (err) {
      setErrorMsg(`Failed to decode audio: ${err}`);
      setState('error');
    }
  }, [getAudioContext]);

  const handleSeparate = useCallback(async () => {
    if (!audioBuffer) return;
    
    setState('loading-model');
    setErrorMsg('');
    setResults([]);
    setMidiData(null);
    setProgress({
      modelProgress: 0,
      modelMessage: 'Starting download...',
      separationProgress: null,
      transcriptionProgress: null,
    });

    try {
      const webgpu = await checkWebGPUAvailability();
      setWebgpuAvailable(webgpu);

      const { session, provider } = await createSession(selectedModel, webgpu);

      setState('processing');
      setProgress(prev => ({
        ...prev,
        modelProgress: 100,
        modelMessage: `Model loaded! Using ${provider.toUpperCase()}. Starting separation...`,
      }));

      const separationResults = await separateAudio(
        audioBuffer,
        selectedModel,
        session,
        (sepProgress) => {
          setProgress(prev => ({
            ...prev,
            separationProgress: sepProgress,
          }));
        }
      );

      setResults(separationResults);
      setState('done');
    } catch (err) {
      console.error('Separation failed:', err);
      setErrorMsg(`Separation failed: ${err instanceof Error ? err.message : String(err)}`);
      setState('error');
    }
  }, [audioBuffer, selectedModel]);

  const handleTranscribe = useCallback(async () => {
    const vocalsResult = results.find(r => r.name === 'Vocals');
    if (!vocalsResult) {
      setErrorMsg('Vocals not found. Please separate audio first.');
      return;
    }

    setIsTranscribing(true);
    setMidiData(null);
    setErrorMsg('');

    try {
      const midi = await transcribeAudio(
        vocalsResult.audioBuffer,
        (transcriptionProgress) => {
          setProgress(prev => ({
            ...prev,
            transcriptionProgress,
          }));
        }
      );

      setMidiData(midi);
    } catch (err) {
      console.error('Transcription failed:', err);
      setErrorMsg(`Transcription failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsTranscribing(false);
    }
  }, [results]);

  const handleReset = useCallback(() => {
    setState('idle');
    setAudioFile(null);
    setAudioBuffer(null);
    setResults([]);
    setMidiData(null);
    setIsTranscribing(false);
    setErrorMsg('');
    setProgress({
      modelProgress: 0,
      modelMessage: '',
      separationProgress: null,
      transcriptionProgress: null,
    });
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-indigo-950 text-white">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-2 text-center">🎵 Audio Separator</h1>
        <p className="text-center text-gray-400 mb-8">
          Demucs HT — Browser-optimized vocal separation + MIDI transcription
        </p>

        {state === 'idle' && (
          <div className="bg-white/5 rounded-xl p-8 border border-white/10">
            <h2 className="text-2xl font-semibold mb-4">Upload Audio File</h2>
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              className="w-full p-4 bg-white/10 rounded-lg border border-white/20 cursor-pointer hover:bg-white/15 transition"
            />
          </div>
        )}

        {state === 'file-loaded' && audioFile && (
          <div className="bg-white/5 rounded-xl p-8 border border-white/10 space-y-4">
            <h2 className="text-2xl font-semibold">File: {audioFile.name}</h2>
            <p className="text-gray-400">Size: {(audioFile.size / 1024 / 1024).toFixed(2)} MB</p>
            
            <div>
              <label className="block text-sm font-medium mb-2">Select Model:</label>
              <select
                value={selectedModel.id}
                onChange={(e) => setSelectedModel(MODELS.find(m => m.id === e.target.value) || MODELS[0])}
                className="w-full p-3 bg-white/10 rounded-lg border border-white/20"
              >
                {MODELS.map(model => (
                  <option key={model.id} value={model.id} className="bg-gray-900">
                    {model.name} ({model.size})
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleSeparate}
              className="w-full py-4 bg-indigo-600 hover:bg-indigo-700 rounded-lg font-semibold transition"
            >
              🎛️ Separate Vocals & Instrumental
            </button>
          </div>
        )}

        {(state === 'loading-model' || state === 'processing') && (
          <div className="bg-white/5 rounded-xl p-8 border border-white/10 space-y-4">
            <h2 className="text-2xl font-semibold">Processing...</h2>
            
            <div>
              <div className="flex justify-between mb-2">
                <span>{progress.modelMessage}</span>
                <span>{Math.round(progress.modelProgress)}%</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2">
                <div
                  className="bg-indigo-600 h-2 rounded-full transition-all"
                  style={{ width: `${progress.modelProgress}%` }}
                />
              </div>
            </div>

            {progress.separationProgress && (
              <div>
                <div className="flex justify-between mb-2">
                  <span>{progress.separationProgress.message}</span>
                  <span>{Math.round(progress.separationProgress.progress)}%</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div
                    className="bg-purple-600 h-2 rounded-full transition-all"
                    style={{ width: `${progress.separationProgress.progress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {state === 'done' && results.length > 0 && (
          <div className="bg-white/5 rounded-xl p-8 border border-white/10 space-y-6">
            <h2 className="text-2xl font-semibold">✅ Separation Complete!</h2>

            {results.map((result, idx) => (
              <div key={idx} className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-xl font-semibold">{result.name}</h3>
                  <div className="flex gap-2">
                    {result.name === 'Vocals' && (
                      <button
                        onClick={handleTranscribe}
                        disabled={isTranscribing}
                        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg font-medium transition"
                      >
                        {isTranscribing ? '🎼 Transcribing...' : '🎼 Create MIDI'}
                      </button>
                    )}
                    <button
                      onClick={() => {
                        const blob = new Blob([result.wavData], { type: 'audio/wav' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${audioFile?.name.replace(/\.[^.]+$/, '') || 'audio'}_${result.name.toLowerCase()}.wav`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                      className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition"
                    >
                      ⬇️ Download WAV
                    </button>
                  </div>
                </div>
                <audio controls className="w-full">
                  <source src={URL.createObjectURL(new Blob([result.wavData], { type: 'audio/wav' }))} type="audio/wav" />
                </audio>
              </div>
            ))}

            {progress.transcriptionProgress && (
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="flex justify-between mb-2">
                  <span>{progress.transcriptionProgress.message}</span>
                  <span>{Math.round(progress.transcriptionProgress.progress)}%</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div
                    className="bg-purple-600 h-2 rounded-full transition-all"
                    style={{ width: `${progress.transcriptionProgress.progress}%` }}
                  />
                </div>
              </div>
            )}

            {midiData && (
              <div className="bg-purple-500/20 rounded-lg p-4 border border-purple-500/30">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-xl font-semibold">🎼 MIDI Ready!</h3>
                    <p className="text-gray-400">Size: {(midiData.length / 1024).toFixed(2)} KB</p>
                  </div>
                  <button
                    onClick={() => downloadMidi(midiData, `${audioFile?.name.replace(/\.[^.]+$/, '') || 'audio'}_vocals.mid`)}
                    className="px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold transition"
                  >
                    ⬇️ Download MIDI
                  </button>
                </div>
              </div>
            )}

            <button
              onClick={handleReset}
              className="w-full py-3 bg-white/10 hover:bg-white/15 rounded-lg font-medium transition"
            >
              🔄 Start Over
            </button>
          </div>
        )}

        {state === 'error' && (
          <div className="bg-red-500/20 rounded-xl p-8 border border-red-500/30">
            <h2 className="text-2xl font-semibold text-red-400 mb-4">⚠️ Error</h2>
            <p className="mb-4">{errorMsg}</p>
            <button
              onClick={handleReset}
              className="w-full py-3 bg-white/10 hover:bg-white/15 rounded-lg font-medium transition"
            >
              🔄 Try Again
            </button>
          </div>
        )}

        <div className="mt-8 text-center text-sm text-gray-500">
          <p>🔒 100% Private: All processing happens in your browser</p>
          <p>🌐 Auto-mirrors: Tries multiple download sources automatically</p>
        </div>
      </div>
    </div>
  );
}
