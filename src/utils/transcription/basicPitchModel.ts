/**
 * Basic Pitch ONNX model loader and inference
 * Spotify's lightweight audio-to-MIDI model (~230 KB)
 */
import * as ort from 'onnxruntime-web';

const MODEL_URL = 'https://huggingface.co/AEmotionStudio/basic-pitch-onnx-models/resolve/main/nmp.onnx';

let session: ort.InferenceSession | null = null;

export async function loadBasicPitchModel(): Promise<ort.InferenceSession> {
  if (session) return session;

  console.log('[BasicPitch] Loading model...');
  
  // Check cache first
  const cache = await caches.open('basic-pitch-model');
  const cached = await cache.match(MODEL_URL);
  
  let modelBuffer: ArrayBuffer;
  
  if (cached) {
    console.log('[BasicPitch] Model loaded from cache');
    modelBuffer = await cached.arrayBuffer();
  } else {
    console.log('[BasicPitch] Downloading model...');
    const response = await fetch(MODEL_URL);
    if (!response.ok) {
      throw new Error(`Failed to download Basic Pitch model: ${response.status}`);
    }
    modelBuffer = await response.arrayBuffer();
    
    // Cache for future use
    await cache.put(MODEL_URL, new Response(modelBuffer));
    console.log('[BasicPitch] Model cached');
  }
  
  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  
  console.log('[BasicPitch] Model ready');
  return session;
}

export interface BasicPitchOutput {
  contour: Float32Array;  // [1, 360, 1, 3]
  note: Float32Array;     // [1, 360, 1, 88]
  onset: Float32Array;    // [1, 360, 1, 88]
}

export async function runBasicPitchInference(
  audioWindow: Float32Array  // 43844 samples @ 22050 Hz
): Promise<BasicPitchOutput> {
  if (!session) {
    throw new Error('Model not loaded. Call loadBasicPitchModel() first.');
  }
  
  // Convert audio to spectrogram-like representation
  // Basic Pitch expects input shape [1, 2281, 1, 1]
  // We need to compute a simple magnitude spectrum
  const fftSize = 2048;
  const hopSize = 512;
  const numFrames = Math.floor((audioWindow.length - fftSize) / hopSize) + 1;
  const numBins = 228; // Frequency bins
  
  // Simple STFT magnitude
  const spectrogram = new Float32Array(numFrames * numBins);
  
  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * hopSize;
    const frameData = audioWindow.slice(start, start + fftSize);
    
    // Apply Hann window
    const windowed = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      windowed[i] = frameData[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / fftSize));
    }
    
    // Simple FFT (magnitude only)
    const fftResult = simpleFFT(windowed);
    
    // Take first numBins frequency bins
    for (let bin = 0; bin < numBins; bin++) {
      spectrogram[frame * numBins + bin] = fftResult[bin];
    }
  }
  
  // Reshape to [1, 2281, 1, 1] - flatten spectrogram
  // 2281 = numFrames * numBins (approximately)
  const inputData = new Float32Array(2281);
  for (let i = 0; i < Math.min(spectrogram.length, 2281); i++) {
    inputData[i] = spectrogram[i];
  }
  
  // Prepare input tensor
  const inputTensor = new ort.Tensor('float32', inputData, [1, 2281, 1, 1]);
  
  // Run inference
  const feeds: Record<string, ort.Tensor> = {
    'input_1': inputTensor
  };
  
  const results = await session.run(feeds);
  
  // Extract outputs
  const contour = results['contour'].data as Float32Array;
  const note = results['note'].data as Float32Array;
  const onset = results['onset'].data as Float32Array;
  
  return { contour, note, onset };
}

// Simple FFT implementation (magnitude only)
function simpleFFT(signal: Float32Array): Float32Array {
  const N = signal.length;
  const magnitudes = new Float32Array(N / 2);
  
  // DFT (simplified, not optimized)
  for (let k = 0; k < N / 2; k++) {
    let real = 0;
    let imag = 0;
    
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      real += signal[n] * Math.cos(angle);
      imag -= signal[n] * Math.sin(angle);
    }
    
    magnitudes[k] = Math.sqrt(real * real + imag * imag) / N;
  }
  
  return magnitudes;
}

export const BASIC_PITCH_CONFIG = {
  SAMPLE_RATE: 22050,
  WINDOW_SIZE: 43844,  // 2 seconds at 22050 Hz
  OVERLAP: 7680,       // ~0.35 seconds overlap
  NUM_FREQ_BINS: 88,   // MIDI notes 21-108
  NUM_FRAMES: 360,     // Time frames per window
};
