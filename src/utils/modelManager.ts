/**
 * Model manager: handles downloading, caching, and loading ONNX models
 * Using UVR-MDX-NET models from GitHub Releases - works in Russia without VPN
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
  chunkSize: number; // samples per chunk
  overlap: number;
  // Model I/O
  inputName: string;
  outputName: string;
}

// GitHub Releases URL - works in Russia without VPN
const GITHUB_BASE = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models';

export const MODELS: ModelConfig[] = [
  {
    id: 'uvr-mdxnet-9482',
    name: 'UVR-MDX-NET 9482 (Fast)',
    description: 'Fast vocal separation. 28MB. Good balance of speed and quality.',
    url: `${GITHUB_BASE}/UVR_MDXNET_9482.onnx`,
    size: '~28 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    nFft: 768,
    hopLength: 384,
    winLength: 768,
    chunkSize: 262144, // ~6 seconds
    overlap: 2,
    inputName: 'input',
    outputName: 'output',
  },
  {
    id: 'uvr-mdxnet-voc-ft',
    name: 'UVR-MDX-NET Voc_FT (Best Quality)',
    description: 'High-quality vocal separation. 64MB. Best results.',
    url: `${GITHUB_BASE}/UVR-MDX-NET-Voc_FT.onnx`,
    size: '~64 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    nFft: 1024,
    hopLength: 512,
    winLength: 1024,
    chunkSize: 262144, // ~6 seconds
    overlap: 2,
    inputName: 'input',
    outputName: 'output',
  },
  {
    id: 'uvr-mdxnet-inst-hq4',
    name: 'UVR-MDX-NET Inst_HQ_4 (Alternative)',
    description: 'Alternative high-quality model. 56MB. Different training.',
    url: `${GITHUB_BASE}/UVR-MDX-NET-Inst_HQ_4.onnx`,
    size: '~56 MB',
    stems: ['Vocals', 'Instrumental'],
    sampleRate: 44100,
    nFft: 1024,
    hopLength: 512,
    winLength: 1024,
    chunkSize: 262144, // ~6 seconds
    overlap: 2,
    inputName: 'input',
    outputName: 'output',
  },
];

const CACHE_NAME = 'audio-separator-models-v3';

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
