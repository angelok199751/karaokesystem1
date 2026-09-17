/**
 * Model manager: handles downloading, caching, and loading ONNX models
 * Using bgkb/bs_polarformer - a real working model for browser-based vocal separation
 */
import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  id: string;
  name: string;
  description: string;
  wasmUrl: string;
  webgpuUrl: string;
  size: string;
  stems: string[];
  // Audio processing params
  sampleRate: number;
  nFft: number;
  hopLength: number;
  winLength: number;
  chunkSize: number; // samples per chunk
  overlap: number;
  // Model I/O
  inputName: string;
  outputName: string;
}

const HF_BASE = 'https://huggingface.co/bgkb/bs_polarformer/resolve/main';

export const MODELS: ModelConfig[] = [
  {
    id: 'bs-polarformer-fp16',
    name: 'BS PolarFormer FP16 (Recommended)',
    description: 'High-quality vocal separation. 103MB. Best balance of quality and speed.',
    wasmUrl: `${HF_BASE}/bs_polarformer_fp16.onnx`,
    webgpuUrl: `${HF_BASE}/bs_polarformer_webgpu_fp16.onnx`,
    size: '~103 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    nFft: 2048,
    hopLength: 512,
    winLength: 2048,
    chunkSize: 131072,
    overlap: 2,
    inputName: 'stft_features',
    outputName: 'mask',
  },
  {
    id: 'bs-polarformer-fp32',
    name: 'BS PolarFormer FP32 (Highest Quality)',
    description: 'Maximum quality vocal separation. 201MB. Slower download but best results.',
    wasmUrl: `${HF_BASE}/bs_polarformer.onnx`,
    webgpuUrl: `${HF_BASE}/bs_polarformer_webgpu.onnx`,
    size: '~201 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    nFft: 2048,
    hopLength: 512,
    winLength: 2048,
    chunkSize: 131072,
    overlap: 2,
    inputName: 'stft_features',
    outputName: 'mask',
  },
];

const CACHE_NAME = 'audio-separator-models-v2';

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

  // Download with progress tracking
  const response = await fetch(url, {
    mode: 'cors',
    credentials: 'omit',
  });

  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.status} ${response.statusText}`);
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
  const providers = useWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
  let lastError: Error | null = null;

  for (const provider of providers) {
    try {
      const modelUrl = provider === 'webgpu' ? modelConfig.webgpuUrl : modelConfig.wasmUrl;
      
      // Download model
      const modelBuffer = await downloadModel(modelUrl);
      
      const opts: ort.InferenceSession.SessionOptions = {
        executionProviders: [provider as unknown as ort.InferenceSession.ExecutionProviderConfig],
        graphOptimizationLevel: 'all',
      };

      if (provider === 'webgpu') {
        // Configure WebGPU
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
