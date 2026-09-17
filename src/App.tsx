import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { separateAudio, SeparationResult, SeparationProgress } from './utils/separation';
import { AudioPlayer } from './components/AudioPlayer';
import { FileUpload } from './components/FileUpload';
import { ProgressBar } from './components/ProgressBar';

type AppState = 'idle' | 'file-loaded' | 'processing' | 'done' | 'error';

interface AppProgress {
  modelProgress: number;
  modelMessage: string;
  separationProgress: SeparationProgress | null;
}

function App() {
  const [state, setState] = useState<AppState>('idle');
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [results, setResults] = useState<SeparationResult[]>([]);
  const [progress, setProgress] = useState<AppProgress>({
    modelProgress: 100,
    modelMessage: 'No model needed! Using classical signal processing.',
    separationProgress: null,
  });
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [originalUrl, setOriginalUrl] = useState<string>('');
  
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
    
    setState('processing');
    setErrorMsg('');
    setResults([]);
    setProgress({
      modelProgress: 100,
      modelMessage: 'Processing audio using classical signal processing...',
      separationProgress: null,
    });

    try {
      // Separate audio using classical methods (no model needed!)
      const separationResults = await separateAudio(
        audioBuffer,
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
  }, [audioBuffer]);

  const handleReset = useCallback(() => {
    setState('idle');
    setAudioFile(null);
    setAudioBuffer(null);
    setResults([]);
    setErrorMsg('');
    setProgress({
      modelProgress: 100,
      modelMessage: 'No model needed! Using classical signal processing.',
      separationProgress: null,
    });
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
              <p className="text-xs text-gray-400">Split vocals & instrumentals in your browser</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Info banner */}
        <div className="mb-8 p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
          <p className="text-sm text-indigo-200">
            <strong>🔒 100% Private:</strong> All processing happens in your browser. Audio files are never uploaded to any server.
            Uses <strong>classical signal processing</strong> — no ML model needed, works instantly!
          </p>
        </div>

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
              🎛️ Separate Vocals
            </button>
          </div>
        )}

        {/* Processing */}
        {state === 'processing' && (
          <div className="space-y-6">
            <ProgressBar
              progress={progress.modelProgress}
              message={progress.modelMessage}
              variant="model"
            />
            
            {progress.separationProgress && (
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
                  Processing audio using Center Channel Extraction method...
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
            {results.map((result, idx) => (
              <div key={idx} className="p-4 rounded-xl bg-white/5 border border-white/10">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">
                      {result.stemName === 'Vocals' ? '🎤' : '🎸'}
                    </span>
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
                  color={idx === 0 ? 'pink' : 'blue'}
                />
              </div>
            ))}
          </div>
        )}

        {/* Error */}
        {state === 'error' && (
          <div className="space-y-4">
            <div className="p-6 rounded-xl bg-red-500/10 border border-red-500/20">
              <h3 className="text-lg font-medium text-red-400 mb-2">⚠️ Error</h3>
              <p className="text-sm text-red-300">{errorMsg}</p>
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
              <h4 className="font-medium text-gray-300 mb-1">⚡ Instant Processing</h4>
              <p>No model download needed. Works instantly on any audio file.</p>
            </div>
            <div className="p-4 rounded-lg bg-white/5">
              <h4 className="font-medium text-gray-300 mb-1">🎯 How It Works</h4>
              <p>Uses Center Channel Extraction - vocals are typically panned to center in stereo.</p>
            </div>
            <div className="p-4 rounded-lg bg-white/5">
              <h4 className="font-medium text-gray-300 mb-1">🎵 Best For</h4>
              <p>Stereo audio with centered vocals. Works great for karaoke creation.</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
