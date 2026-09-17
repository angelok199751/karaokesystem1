/**
 * Model manager for demucs-web
 */
import { DemucsProcessor } from 'demucs-web';
import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  id: string;
  name: string;
  description: string;
  url: string;
  size: string;
  stems: string[];
  sampleRate: number;
  chunkSamples: number;
}

const HF_BASE = 'https://huggingface.co';

export const MODELS: ModelConfig[] = [
  {
    id: 'demucs-web-htdemucs',
    name: 'Demucs HT (Browser Optimized)',
    description: 'Optimized for browser. 170MB. Best balance of quality and performance.',
    url: `${HF_BASE}/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx`,
    size: '~170 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    chunkSamples: 343980,
  },
];

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

export async function downloadModel(
  url: string,
  onProgress?: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  const urlsToTry: string[] = [];
  
  if (url.includes('huggingface.co')) {
    urlsToTry.push(url.replace('https://huggingface.co', 'https://hf-mirror.com'));
    urlsToTry.push(url);
    urlsToTry.push(url.replace('https://huggingface.co', 'https://hub.nuaa.cf'));
  } else {
    urlsToTry.push(url);
  }

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

      return result.buffer;
    } catch (e) {
      console.warn(`Failed to download from ${tryUrl}:`, e);
    }
  }

  throw new Error('Failed to download model from all sources');
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
        graphOptimizationLevel: 'basic',
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
