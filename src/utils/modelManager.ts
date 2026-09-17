/**
 * Model manager: handles downloading, caching, and loading ONNX models
 */
import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  id: string;
  name: string;
  description: string;
  url: string;
  size: string;
  stems: string[];
  nFft: number;
  hopLength: number;
  sampleRate: number;
  chunkDuration: number; // seconds
}

export const MODELS: ModelConfig[] = [
  {
    id: 'mdx-net-vocals',
    name: 'MDX-Net Vocals (Fast)',
    description: 'Lightweight model for vocal/instrumental separation. ~30MB. Good quality, fast inference.',
    url: 'https://huggingface.co/joao-matias/mdxnet-vocals-onnx/resolve/main/model.onnx',
    size: '~30 MB',
    stems: ['Vocals', 'Instrumental'],
    nFft: 1024,
    hopLength: 512,
    sampleRate: 44100,
    chunkDuration: 10,
  },
  {
    id: 'mdx-net-kim',
    name: 'Kim Vocal 2 (Balanced)',
    description: 'Kim_Vocal_2 model for high-quality vocal separation. ~67MB. Better quality than fast model.',
    url: 'https://huggingface.co/joao-matias/kim-vocal-2-onnx/resolve/main/model.onnx',
    size: '~67 MB',
    stems: ['Vocals', 'Instrumental'],
    nFft: 768,
    hopLength: 384,
    sampleRate: 44100,
    chunkDuration: 10,
  },
  {
    id: 'demucs-htdemucs',
    name: 'HTDemucs (4 stems)',
    description: 'Demucs-style 4-stem separation: drums, bass, other, vocals. ~80MB.',
    url: 'https://huggingface.co/joao-matias/htdemucs-onnx/resolve/main/model.onnx',
    size: '~80 MB',
    stems: ['Drums', 'Bass', 'Other', 'Vocals'],
    nFft: 4096,
    hopLength: 1024,
    sampleRate: 44100,
    chunkDuration: 10,
  },
];

const CACHE_NAME = 'audio-separator-models-v1';

export async function getCachedModel(modelId: string): Promise<ArrayBuffer | null> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(modelId);
    if (response) {
      return await response.arrayBuffer();
    }
  } catch (e) {
    console.warn('Cache read failed:', e);
  }
  return null;
}

export async function cacheModel(modelId: string, data: ArrayBuffer): Promise<void> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = new Response(data, {
      headers: { 'Content-Type': 'application/octet-stream' },
    });
    await cache.put(modelId, response);
  } catch (e) {
    console.warn('Cache write failed:', e);
  }
}

export async function downloadModel(
  config: ModelConfig,
  onProgress?: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  // Check cache first
  const cached = await getCachedModel(config.id);
  if (cached) {
    onProgress?.(cached.byteLength, cached.byteLength);
    return cached;
  }

  // Download with progress tracking
  const response = await fetch(config.url, {
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
  await cacheModel(config.id, result.buffer);

  return result.buffer;
}

export async function createSession(
  modelBuffer: ArrayBuffer,
  useWebGPU: boolean = true
): Promise<ort.InferenceSession> {
  // Try providers in order of preference
  const providers: string[] = [];
  
  if (useWebGPU) {
    try {
      const gpuAvailable = await checkWebGPUAvailability();
      if (gpuAvailable) {
        providers.push('webgpu');
      }
    } catch {
      // WebGPU not available
    }
  }

  // Fallback to WASM (single-threaded for GitHub Pages compatibility)
  providers.push('wasm');

  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: providers as any,
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
  });

  return session;
}

export async function checkWebGPUAvailability(): Promise<boolean> {
  try {
    const nav = navigator as any;
    if (!nav.gpu) return false;
    const adapter = await nav.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}
