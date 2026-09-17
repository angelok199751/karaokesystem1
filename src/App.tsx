import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import * as ort from 'onnxruntime-web';
import { MODELS, ModelConfig, checkWebGPUAvailability } from './utils/modelManager';
import { separateAudio, SeparationResult, SeparationProgress } from './utils/separation';
import { AudioPlayer } from './components/AudioPlayer';
import { FileUpload } from './components/FileUpload';
import { ProgressBar } from './components/ProgressBar';
import { ModelSelector } from './components/ModelSelector';

type AppState = 'idle' | 'file-loaded' | 'loading-model' | 'processing' | 'done' | 'error';

interface AppProgress {
  modelProgress: number;
  modelMessage: string;
  separationProgress: SeparationProgress | null;
}

function App() {
  const [state, setState] = useState<AppState>('idle');
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelConfig>(MODELS[0]);
  const [results, setResults] = useState<SeparationResult[]>([]);
  const [progress, setProgress] = useState<AppProgress>({
    modelProgress: 0,
    modelMessage: '',
    separationProgress: null,
  });
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [webgpuAvailable, setWebgpuAvailable] = useState<boolean>(false);
  const [originalUrl, setOriginalUrl] = useState<string>('');
  const [activeProvider, setActiveProvider] = useState<string>('');
  
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Configure onnxruntime-web on mount
  useEffect(() => {
    checkWebGPUAvailability().then(setWebgpuAvailable);
    
    // Load WASM from CDN — version MUST match installed npm package (1.21.0)
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';
    ort.env.logLevel = 'warning';
  }, []);

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
    setErrorMsg('');
    
    try {
      const ctx = getAudioContext();
      const arrayBuffer = await file.arrayBuffer();
      const buffer = await ctx.decodeAudioData(arrayBuffer);
      setAudioBuffer(buffer);
      
      const url = URL.createObjectURL(file);
      setOriginalUrl(url);
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
    setActiveProvider('');
    setProgress({
      modelProgress: 0,
      modelMessage: 'Starting download...',
      separationProgress: null,
    });

    try {
      // Download model (demucs-web will load it internally)
      const { downloadModel } = await import('./utils/modelManager');
      await downloadModel(selectedModel.url, (loaded, total) => {
        const progress = total > 0 ? (loaded / total) * 100 : 0;
        setProgress(prev => ({
          ...prev,
          modelProgress: progress,
          modelMessage: `Downloading model... ${Math.round(progress)}%`,
        }));
      });

      setState('processing');
      setProgress(prev => ({
        ...prev,
        modelProgress: 100,
        modelMessage: `Model loaded! Starting separation...`,
      }));
      setActiveProvider('demucs-web');

      // Separate audio using demucs-web
      const separationResults = await separateAudio(
        audioBuffer,
        selectedModel,
        null as any, // session not needed, demucs-web manages it
        (sepProgress: SeparationProgress) => {
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
  }, [audioBuffer, selectedModel, webgpuAvailable]);

  const handleReset = useCallback(() => {
    setState('idle');
    setAudioFile(null);
    setAudioBuffer(null);
    setResults([]);
    setErrorMsg('');
    setActiveProvider('');
    setProgress({ modelProgress: 0, modelMessage: '', separationProgress: null });
    if (originalUrl) {
      URL.revokeObjectURL(originalUrl);
      setOriginalUrl('');
    }
  }, [originalUrl]);

  // Memoize blob URLs for results
  const resultUrls = useMemo(() => {
    return results.map(result => {
      const blob = new Blob([result.wavData], { type: 'audio/wav' });
      return URL.createObjectURL(blob);
    });
  }, [results]);
  
  useEffect(() => {
    return () => {
      resultUrls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [resultUrls]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-indigo-950">
      {/* Header */}
      <header className="border-b border-white/10 backdrop-blur-sm bg-black/20">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-xl">
              🎵
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Audio Separator</h1>
              <p className="text-xs text-gray-400">Demucs HT — Browser-optimized vocal separation</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
              webgpuAvailable 
                ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
            }`}>
              <span className={`w-2 h-2 rounded-full ${webgpuAvailable ? 'bg-green-400' : 'bg-yellow-400'}`}></span>
              {webgpuAvailable ? 'WebGPU' : 'WASM'}
            </span>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Info banner */}
        <div className="mb-8 p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
          <p className="text-sm text-indigo-200">
            <strong>🔒 100% Private:</strong> All processing happens in your browser. Audio files are never uploaded to any server. 
            Models are cached locally after first download. Uses <strong>Demucs HT</strong> — browser-optimized vocal separation.
          </p>
        </div>

        {/* Model selector */}
        <ModelSelector
          models={MODELS}
          selected={selectedModel}
          onChange={setSelectedModel}
          disabled={state === 'loading-model' || state === 'processing'}
        />

        {/* File upload */}
        {state === 'idle' && (
          <FileUpload onFileSelect={handleFileSelect} />
        )}

        {/* File loaded */}
        {state === 'file-loaded' && audioFile && audioBuffer && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-white/5 border border-white/10">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-violet-500/20 flex items-center justify-center text-2xl">
                  🎶
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium text-white truncate">{audioFile.name}</h3>
                  <p className="text-sm text-gray-400">
                    {(audioFile.size / 1024 / 1024).toFixed(2)} MB • {audioBuffer.duration.toFixed(1)}s • {audioBuffer.sampleRate}Hz • {audioBuffer.numberOfChannels}ch
                  </p>
                </div>
                <button
                  onClick={handleReset}
                  className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 text-sm transition-colors"
                >
                  Change file
                </button>
              </div>
              
              {originalUrl && (
                <div className="mt-4">
                  <AudioPlayer audioUrl={originalUrl} label="Original" color="violet" />
                </div>
              )}
            </div>

            <button
              onClick={handleSeparate}
              className="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-semibold text-lg transition-all shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 active:scale-[0.98]"
            >
              🎛️ Separate Vocals & Instrumental
            </button>
          </div>
        )}

        {/* Loading / Processing */}
        {(state === 'loading-model' || state === 'processing') && (
          <div className="space-y-6">
            <ProgressBar
              progress={state === 'loading-model' ? progress.modelProgress : 100}
              message={state === 'loading-model' ? progress.modelMessage : `Model loaded! Using ${activeProvider.toUpperCase()}`}
              variant="model"
            />
            
            {state === 'processing' && progress.separationProgress && (
              <ProgressBar
                progress={progress.separationProgress.progress}
                message={progress.separationProgress.message}
                variant="separation"
              />
            )}

            <div className="p-4 rounded-xl bg-white/5 border border-white/10">
              <div className="flex items-center gap-3">
                <div className="animate-spin w-5 h-5 border-2 border-violet-400 border-t-transparent rounded-full"></div>
                <p className="text-sm text-gray-300">
                  {state === 'loading-model' 
                    ? 'Downloading model... First time may take a while (336MB). The model will be cached for future use.'
                    : 'Processing audio through the neural network. This may take a few minutes depending on file length and hardware.'
                  }
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {state === 'done' && results.length > 0 && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white">🎉 Separation Complete!</h2>
              <button
                onClick={handleReset}
                className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 text-sm transition-colors border border-white/10"
              >
                Start Over
              </button>
            </div>

            {/* Original */}
            {originalUrl && (
              <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                <AudioPlayer audioUrl={originalUrl} label="Original" color="violet" />
              </div>
            )}

            {/* Stems */}
            {results.map((result, idx) => {
              const colors: Array<'pink' | 'blue'> = ['pink', 'blue'];
              const icons: Record<string, string> = {
                'Vocals': '🎤',
                'Instrumental': '🎸',
              };
              
              return (
                <div key={idx} className="p-4 rounded-xl bg-white/5 border border-white/10">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{icons[result.stemName] || '🎵'}</span>
                      <span className="font-medium text-white">{result.stemName}</span>
                    </div>
                    <button
                      onClick={() => {
                        const blob = new Blob([result.wavData], { type: 'audio/wav' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${audioFile?.name?.replace(/\.[^.]+$/, '') || 'audio'}_${result.stemName.toLowerCase()}.wav`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                      className="px-3 py-1.5 rounded-lg bg-green-500/20 hover:bg-green-500/30 text-green-400 text-sm font-medium transition-colors border border-green-500/30"
                    >
                      ⬇️ Download WAV
                    </button>
                  </div>
                  <AudioPlayer 
                    audioUrl={resultUrls[idx]} 
                    label={result.stemName}
                    color={colors[idx % colors.length]}
                  />
                </div>
              );
            })}
          </div>
        )}

        {/* Error */}
        {state === 'error' && (
          <div className="space-y-4">
            <div className="p-6 rounded-xl bg-red-500/10 border border-red-500/20">
              <h3 className="text-lg font-medium text-red-400 mb-2">⚠️ Error</h3>
              <p className="text-sm text-red-300">{errorMsg}</p>
              <p className="text-xs text-red-400 mt-2">
                💡 Tip: If model download fails, try using a VPN or wait and retry. 
                The model will be cached after first successful download.
              </p>
            </div>
            <button
              onClick={handleReset}
              className="w-full py-3 px-6 rounded-xl bg-white/5 hover:bg-white/10 text-white font-medium transition-colors border border-white/10"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 pt-8 border-t border-white/10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-400">
            <div className="p-4 rounded-lg bg-white/5">
              <h4 className="font-medium text-gray-300 mb-1">🌐 Browser Support</h4>
              <p>Chrome 113+, Edge 113+, Safari 18+, Firefox 121+ (WebGPU). WASM works everywhere.</p>
            </div>
            <div className="p-4 rounded-lg bg-white/5">
              <h4 className="font-medium text-gray-300 mb-1">⚡ Performance</h4>
              <p>WebGPU is 3-5x faster than WASM. First model download: {selectedModel.size}.</p>
            </div>
            <div className="p-4 rounded-lg bg-white/5">
              <h4 className="font-medium text-gray-300 mb-1">🧠 Model</h4>
              <p>Demucs HT — 170 MB. Browser-optimized. MIT license.</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
