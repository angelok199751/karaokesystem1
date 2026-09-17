/**
 * Basic Pitch ONNX model loader and inference
 * Spotify's lightweight audio-to-MIDI model (~230 KB)
 */
import * as ort from 'onnxruntime-web';

const MODEL_URL = 'https://huggingface.co/AEmotionStudio/basic-pitch-onnx-models/resolve/main/model.onnx';

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
  
  // Prepare input tensor
  const inputTensor = new ort.Tensor('float32', audioWindow, [1, 1, 2281, 1]);
  
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

export const BASIC_PITCH_CONFIG = {
  SAMPLE_RATE: 22050,
  WINDOW_SIZE: 43844,  // 2 seconds at 22050 Hz
  OVERLAP: 7680,       // ~0.35 seconds overlap
  NUM_FREQ_BINS: 88,   // MIDI notes 21-108
  NUM_FRAMES: 360,     // Time frames per window
};
