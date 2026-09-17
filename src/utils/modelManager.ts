/**
 * Model manager for BS-Roformer-SW 6-stem ONNX model
 * Specially prepared for browser inference via onnxruntime-web
 */
import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  id: string;
  name: string;
  description: string;
  url: string;
  size: string;
  stems: string[];
  // Audio processing params
  sampleRate: number;
  nFft: number;
  hopLength: number;
  winLength: number;
  chunkSamples: number; // 176400 = 4s @ 44.1kHz
  chunkFrames: number;  // T = 345
  overlap: number;      // 25% overlap-add
  // Model I/O
  inputNames: [string, string];  // [spec_real, spec_imag]
  outputNames: [string, string]; // [out_spec_real, out_spec_imag]
}

// HuggingFace URLs (try mirror first for Russia)
const HF_MIRROR = 'https://hf-mirror.com';
const HF_BASE = 'https://huggingface.co';

export const MODELS: ModelConfig[] = [
  {
    id: 'bs-roformer-sw-fp16',
    name: 'BS-Roformer-SW FP16 (Recommended)',
    description: 'Vocals + Instrumental separation. 336MB. Best quality/speed.',
    url: `${HF_BASE}/elicwhite/bs-roformer-sw-6stem-onnx/resolve/main/bs_roformer_sw_6stem_fp16.onnx`,
    size: '~336 MB',
    stems: ['Vocals', 'Instrumental'], // Simplified output
    sampleRate: 44100,
    nFft: 2048,
    hopLength: 512,
    winLength: 2048,
    chunkSamples: 176400, // 4s @ 44.1kHz
    chunkFrames: 345,     // T = 345
    overlap: 4,           // 25% overlap
    inputNames: ['spec_real', 'spec_imag'],
    outputNames: ['out_spec_real', 'out_spec_imag'],
  },
  {
    id: 'bs-roformer-sw-fp32',
    name: 'BS-Roformer-SW FP32 (Fallback)',
    description: 'Same model, fp32 weights. 669MB. Use if fp16 fails.',
    url: `${HF_BASE}/elicwhite/bs-roformer-sw-6stem-onnx/resolve/main/bs_roformer_sw_6stem_fp32.onnx`,
    size: '~669 MB',
    stems: ['Vocals', 'Instrumental'], // Simplified output
    sampleRate: 44100,
    nFft: 2048,
    hopLength: 512,
    winLength: 2048,
    chunkSamples: 176400,
    chunkFrames: 345,
    overlap: 4,
    inputNames: ['spec_real', 'spec_imag'],
    outputNames: ['out_spec_real', 'out_spec_imag'],
  },
];

const CACHE_NAME = 'audio-separator-models-v4';

export async function getCachedModel(modelUrl: string): Promise<ArrayBuffer | null> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(modelUrl);
    if (response) {
      return await response.arrayBuffer();
    }
  } catch (e) {
    console.warn('Cache read failed:', e);
  }
  return null;
}

export async function cacheModel(modelUrl: string, modelBuffer: ArrayBuffer): Promise<void> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = new Response(modelBuffer, {
      headers: { 'Content-Type': 'application/octet-stream' },
    });
    await cache.put(modelUrl, response);
  } catch (e) {
    console.warn('Cache write failed:', e);
  }
}

export async function downloadModel(
  url: string,
  onProgress?: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  // Check cache first
  const cached = await getCachedModel(url);
  if (cached) {
    onProgress?.(cached.byteLength, cached.byteLength);
    return cached;
  }

  // Try multiple URLs (mirror + direct)
  const urlsToTry: string[] = [url];
  
  // Add mirror URL
  if (url.includes('huggingface.co') && !url.includes('hf-mirror.com')) {
    const mirrorUrl = url.replace('https://huggingface.co', HF_MIRROR);
    urlsToTry.push(mirrorUrl);
  }

  let lastError: Error | null = null;
  
  for (const tryUrl of urlsToTry) {
    try {
      const response = await fetch(tryUrl, {
        mode: 'cors',
        credentials: 'omit',
        redirect: 'follow',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentLength = response.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength, 10) : 0;

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let loaded = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        onProgress?.(loaded, total);
      }

      // Combine chunks
      const result = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
      }

      // Cache the model
      await cacheModel(url, result.buffer);

      return result.buffer;
    } catch (e) {
      console.warn(`Failed to download from ${tryUrl}:`, e);
      lastError = e instanceof Error ? e : new Error(String(e));
    }
  }

  throw lastError || new Error('Failed to download model from all sources');
}

export async function checkWebGPUAvailability(): Promise<boolean> {
  try {
    const nav = navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } };
    if (!nav.gpu) return false;
    const adapter = await nav.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

export async function createSession(
  modelConfig: ModelConfig,
  useWebGPU: boolean
): Promise<{ session: ort.InferenceSession; provider: string }> {
  // WebGPU is strongly recommended for this model
  const providers = useWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
  let lastError: Error | null = null;

  for (const provider of providers) {
    try {
      // Download model
      const modelBuffer = await downloadModel(modelConfig.url);
      
      const opts: ort.InferenceSession.SessionOptions = {
        executionProviders: [provider as unknown as ort.InferenceSession.ExecutionProviderConfig],
        graphOptimizationLevel: 'all',
      };

      if (provider === 'webgpu') {
        const ortEnv = ort.env as unknown as { webgpu?: { powerPreference?: string } };
        ortEnv.webgpu = ortEnv.webgpu || {};
        ortEnv.webgpu.powerPreference = 'high-performance';
      }

      const session = await ort.InferenceSession.create(modelBuffer, opts);
      return { session, provider };
    } catch (e) {
      console.warn(`Failed to initialize ${provider} backend:`, e);
      lastError = e instanceof Error ? e : new Error(String(e));
    }
  }

  throw lastError || new Error('Failed to create inference session.');
}
