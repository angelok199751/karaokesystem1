/**
 * Basic Pitch ONNX model wrapper
 */
import * as ort from 'onnxruntime-web';

export const BASIC_PITCH_CONFIG = {
  SAMPLE_RATE: 22050,
  WINDOW_SIZE: 43844,
  OVERLAP: 7680,
  NUM_FREQ_BINS: 88,
  FFT_HOP: 256,
};

let session: ort.InferenceSession | null = null;

export async function loadBasicPitchModel(): Promise<void> {
  if (session) return;

  const MODEL_URL = 'https://huggingface.co/AEmotionStudio/basic-pitch-onnx-models/resolve/main/nmp.onnx';
  
  console.log('[BasicPitch] Loading model...');
  
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
    await cache.put(MODEL_URL, new Response(modelBuffer));
    console.log('[BasicPitch] Model cached');
  }
  
  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  
  console.log('[BasicPitch] Model ready');
  console.log('[BasicPitch] Input names:', session.inputNames);
  console.log('[BasicPitch] Output names:', session.outputNames);
}

export interface BasicPitchOutput {
  contour: Float32Array;
  note: Float32Array;
  onset: Float32Array;
}

export async function runBasicPitchInference(
  audioWindow: Float32Array
): Promise<BasicPitchOutput> {
  if (!session) {
    throw new Error('Model not loaded. Call loadBasicPitchModel() first.');
  }
  
  const inputData = new Float32Array(audioWindow.length);
  inputData.set(audioWindow);
  
  const inputTensor = new ort.Tensor('float32', inputData, [1, audioWindow.length, 1]);
  const inputName = session.inputNames[0];
  
  const feeds: Record<string, ort.Tensor> = {
    [inputName]: inputTensor
  };
  
  const results = await session.run(feeds);
  const outputNames = session.outputNames;
  
  if (outputNames.length >= 3) {
    const note = results[outputNames[0]].data as Float32Array;
    const onset = results[outputNames[1]].data as Float32Array;
    const contour = results[outputNames[2]].data as Float32Array;
    
    return { contour, note, onset };
  } else {
    throw new Error(`Unexpected number of outputs: ${outputNames.length}`);
  }
}
