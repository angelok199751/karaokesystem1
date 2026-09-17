/**
 * Model manager for UVR-MDX-NET models
 * Lightweight vocal separation models (28-80 MB)
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
  chunkSamples: number;
  overlap: number;
  // Model I/O
  inputName: string;
  outputName: string;
}

// HuggingFace URLs
const HF_BASE = 'https://huggingface.co';

export const MODELS: ModelConfig[] = [
  {
    id: 'uvr-mdxnet-9482',
    name: 'UVR-MDX-NET 9482 (Fast)',
    description: 'Fast vocal separation. 28MB. Good balance of speed and quality.',
    url: `${HF_BASE}/Blane187/all_public_uvr_models/resolve/main/UVR_MDXNET_9482.onnx`,
    size: '~28 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    nFft: 768,
    hopLength: 384,
    winLength: 768,
    chunkSamples: 262144, // ~6s @ 44.1kHz
    overlap: 2,
    inputName: 'input',
    outputName: 'output',
  },
  {
    id: 'uvr-mdxnet-voc-ft',
    name: 'UVR-MDX-NET Voc_FT (Best Quality)',
    description: 'High-quality vocal separation. 64MB. Best results.',
    url: `${HF_BASE}/Blane187/all_public_uvr_models/resolve/main/UVR-MDX-NET-Voc_FT.onnx`,
    size: '~64 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    nFft: 1024,
    hopLength: 512,
    winLength: 1024,
    chunkSamples: 262144,
    overlap: 2,
    inputName: 'input',
    outputName: 'output',
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

  // Try multiple URLs (direct + mirror)
  const urlsToTry: string[] = [url];
  
  // Add mirror URL
  if (url.includes('huggingface.co') && !url.includes('hf-mirror.com')) {
    const mirrorUrl = url.replace('https://huggingface.co', 'https://hf-mirror.com');
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
