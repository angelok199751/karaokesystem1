/**
 * HT-Demucs FT ONNX model manager
 * Simple audio-to-audio separation (no manual STFT needed!)
 */
import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  id: string;
  name: string;
  description: string;
  url: string;
  size: string;
  stems: string[];
  sampleRate: number;
  chunkSamples: number; // 343980 samples = 7.8s @ 44.1kHz
  inputName: string;
  outputName: string;
}

// HuggingFace URLs
const HF_BASE = 'https://huggingface.co';

export const MODELS: ModelConfig[] = [
  {
    id: 'htdemucs-ft-vocals',
    name: 'HT-Demucs FT Vocals (Best Quality)',
    description: 'State-of-the-art vocal separation. SDR 9.19 dB. 316MB.',
    url: `${HF_BASE}/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals.onnx`,
    size: '~316 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    chunkSamples: 343980, // 7.8s @ 44.1kHz
    inputName: 'mix',
    outputName: 'stems',
  },
];

const CACHE_NAME = 'audio-separator-models-v5';

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

      const result = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
      }

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
