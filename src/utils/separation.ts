/**
 * Audio separation using UVR-MDX-NET ONNX models
 * Adapted for models from GitHub Releases (sherpa-onnx)
 */
import * as ort from 'onnxruntime-web';
import { ModelConfig } from './modelManager';

export interface SeparationResult {
  stemName: string;
  audioBuffer: AudioBuffer;
  wavData: ArrayBuffer;
}

export interface SeparationProgress {
  stage: 'preparing' | 'processing' | 'reconstructing' | 'done';
  progress: number;
  message: string;
  chunk?: number;
  totalChunks?: number;
  elapsed?: number;
  eta?: number;
}

// ── DSP: Hann window, STFT, iSTFT ──────────────────────────────────────────

function hannWindow(len: number): Float32Array {
  const w = new Float32Array(len);
  for (let i = 0; i < len; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / len));
  return w;
}

/** Radix-2 Cooley-Tukey FFT (in-place) */
function fftInPlace(re: Float32Array, im: Float32Array, N: number): void {
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }
  // FFT butterflies
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const angle = -2 * Math.PI / len;
    const wRe = Math.cos(angle), wIm = Math.sin(angle);
    for (let i = 0; i < N; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < half; j++) {
        const a = i + j, b = i + j + half;
        const tRe = curRe * re[b] - curIm * im[b];
        const tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe; im[b] = im[a] - tIm;
        re[a] += tRe; im[a] += tIm;
        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
}

/** Real FFT: returns N/2+1 complex pairs as interleaved [re,im,...] */
function rfft(x: Float32Array, N: number): Float32Array {
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  re.set(x);
  fftInPlace(re, im, N);
  const out = new Float32Array((N / 2 + 1) * 2);
  for (let k = 0; k <= N / 2; k++) {
    out[k * 2] = re[k];
    out[k * 2 + 1] = im[k];
  }
  return out;
}

/** Inverse real FFT: takes N/2+1 complex pairs, returns N real samples */
function irfft(spec: Float32Array, N: number): Float32Array {
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  const half = N / 2;
  for (let k = 0; k <= half; k++) {
    re[k] = spec[k * 2];
    im[k] = -spec[k * 2 + 1];
  }
  for (let k = 1; k < half; k++) {
    re[N - k] = spec[k * 2];
    im[N - k] = spec[k * 2 + 1];
  }
  fftInPlace(re, im, N);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = re[i] / N;
  return out;
}

interface STFTResult {
  data: Float32Array; // [n_freq, n_frames, 2] flattened
  nFrames: number;
}

/** STFT on a mono signal. Returns [n_freq, n_frames, 2] flattened. */
function stft(signal: Float32Array, nFft: number, hop: number, win: Float32Array, nFreq: number): STFTResult {
  const nFrames = Math.floor((signal.length - nFft) / hop) + 1;
  const out = new Float32Array(nFreq * nFrames * 2);
  const windowed = new Float32Array(nFft);
  
  for (let t = 0; t < nFrames; t++) {
    const off = t * hop;
    for (let i = 0; i < nFft; i++) {
      windowed[i] = (signal[off + i] || 0) * win[i];
    }
    const spec = rfft(windowed, nFft);
    for (let f = 0; f < nFreq; f++) {
      out[(f * nFrames + t) * 2] = spec[f * 2];
      out[(f * nFrames + t) * 2 + 1] = spec[f * 2 + 1];
    }
  }
  
  return { data: out, nFrames };
}

/** iSTFT: reconstruct from [n_freq, n_frames, 2] complex STFT. */
function istft(stftData: Float32Array, nFrames: number, nFft: number, hop: number, win: Float32Array, nFreq: number, length: number): Float32Array {
  const out = new Float32Array(length);
  const winSum = new Float32Array(length);
  const spec = new Float32Array(nFreq * 2);

  for (let t = 0; t < nFrames; t++) {
    for (let f = 0; f < nFreq; f++) {
      spec[f * 2] = stftData[(f * nFrames + t) * 2];
      spec[f * 2 + 1] = stftData[(f * nFrames + t) * 2 + 1];
    }
    const frame = irfft(spec, nFft);
    const off = t * hop;
    for (let i = 0; i < nFft && off + i < length; i++) {
      out[off + i] += frame[i] * win[i];
      winSum[off + i] += win[i] * win[i];
    }
  }
  
  for (let i = 0; i < length; i++) {
    if (winSum[i] > 1e-8) out[i] /= winSum[i];
  }
  return out;
}

// ── Prepare model input for UVR-MDX-NET ────────────────────────────────────

interface ChunkInput {
  input: Float32Array;
  nFrames: number;
  stftData: STFTResult;
}

/**
 * Prepare input for UVR-MDX-NET model
 * Input shape: [1, 1, n_freq, n_frames] - magnitude spectrogram
 */
function prepareChunkInput(
  mono: Float32Array,
  win: Float32Array,
  nFft: number,
  hop: number,
  nFreq: number
): ChunkInput {
  const stftResult = stft(mono, nFft, hop, win, nFreq);
  const nFrames = stftResult.nFrames;
  
  // Compute magnitude spectrogram
  const magnitude = new Float32Array(nFreq * nFrames);
  for (let f = 0; f < nFreq; f++) {
    for (let t = 0; t < nFrames; t++) {
      const re = stftResult.data[(f * nFrames + t) * 2];
      const im = stftResult.data[(f * nFrames + t) * 2 + 1];
      magnitude[f * nFrames + t] = Math.sqrt(re * re + im * im);
    }
  }
  
  return { input: magnitude, nFrames, stftData: stftResult };
}

// ── Apply mask and iSTFT ────────────────────────────────────────────────────

interface ReconstructedAudio {
  audio: Float32Array;
}

/**
 * Apply mask from model output and reconstruct audio
 * Model output is a mask in range [0, 1]
 */
function applyMaskAndReconstruct(
  mask: Float32Array,
  stftData: STFTResult,
  nFrames: number,
  win: Float32Array,
  nFft: number,
  hop: number,
  nFreq: number,
  length: number
): ReconstructedAudio {
  const masked = new Float32Array(nFreq * nFrames * 2);

  // Apply mask to complex STFT
  for (let f = 0; f < nFreq; f++) {
    for (let t = 0; t < nFrames; t++) {
      const maskIdx = f * nFrames + t;
      const m = mask[maskIdx];
      
      const stftIdx = (f * nFrames + t) * 2;
      const re = stftData.data[stftIdx];
      const im = stftData.data[stftIdx + 1];
      
      masked[stftIdx] = re * m;
      masked[stftIdx + 1] = im * m;
    }
  }
  
  // Zero DC bin
  for (let t = 0; t < nFrames; t++) {
    masked[t * 2] = 0;
    masked[t * 2 + 1] = 0;
  }

  const audio = istft(masked, nFrames, nFft, hop, win, nFreq, length);
  return { audio };
}

// ── Resample ────────────────────────────────────────────────────────────────

function resample(audioData: Float32Array, fromRate: number, toRate: number): Float32Array<ArrayBuffer> {
  if (fromRate === toRate) {
    const copy = new Float32Array(audioData.length);
    copy.set(audioData);
    return copy;
  }
  const ratio = fromRate / toRate;
  const newLength = Math.round(audioData.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const srcIndex = i * ratio;
    const floor = Math.floor(srcIndex);
    const ceil = Math.min(floor + 1, audioData.length - 1);
    const frac = srcIndex - floor;
    result[i] = audioData[floor] * (1 - frac) + audioData[ceil] * frac;
  }
  return result;
}

// ── Main separation pipeline ────────────────────────────────────────────────

export async function separateAudio(
  audioBuffer: AudioBuffer,
  modelConfig: ModelConfig,
  session: ort.InferenceSession,
  onProgress?: (progress: SeparationProgress) => void
): Promise<SeparationResult[]> {
  const originalSampleRate = audioBuffer.sampleRate;
  const nFreq = modelConfig.nFft / 2 + 1;
  
  onProgress?.({ stage: 'preparing', progress: 0, message: 'Preparing audio...' });

  // Convert to mono
  let mono: Float32Array;
  if (audioBuffer.numberOfChannels === 1) {
    mono = audioBuffer.getChannelData(0);
  } else {
    // Mix down to mono
    const length = audioBuffer.length;
    mono = new Float32Array(length);
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
      const channelData = audioBuffer.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        mono[i] += channelData[i] / audioBuffer.numberOfChannels;
      }
    }
  }

  // Resample to model's sample rate if needed
  const procMono = modelConfig.sampleRate !== originalSampleRate
    ? resample(mono, originalSampleRate, modelConfig.sampleRate)
    : new Float32Array(mono);

  const totalSamples = procMono.length;
  const win = hannWindow(modelConfig.winLength);
  const step = Math.floor(modelConfig.chunkSize / modelConfig.overlap);
  
  // Calculate chunk positions
  const starts: number[] = [];
  for (let s = 0; s < totalSamples; s += step) {
    starts.push(s);
  }

  // Accumulators for overlap-add
  const vocalsAccum = new Float32Array(totalSamples);
  const count = new Float32Array(totalSamples);

  const t0 = performance.now();

  for (let ci = 0; ci < starts.length; ci++) {
    const start = starts[ci];
    const end = Math.min(start + modelConfig.chunkSize, totalSamples);
    const chunkLen = end - start;

    // Extract & pad chunk
    const chunk = new Float32Array(modelConfig.chunkSize);
    chunk.set(procMono.subarray(start, end));

    // STFT & prepare input
    const { input, nFrames, stftData } = prepareChunkInput(
      chunk, win, modelConfig.nFft, modelConfig.hopLength, nFreq
    );

    // Run ONNX inference
    // Input shape: [1, 1, n_freq, n_frames]
    const tensor = new ort.Tensor('float32', input, [1, 1, nFreq, nFrames]);
    const results = await session.run({ [modelConfig.inputName]: tensor });
    const mask = results[modelConfig.outputName].data as Float32Array;

    // Reconstruct
    const recon = applyMaskAndReconstruct(
      mask, stftData, nFrames, win,
      modelConfig.nFft, modelConfig.hopLength, nFreq, modelConfig.chunkSize
    );

    // Accumulate with overlap
    for (let i = 0; i < chunkLen; i++) {
      vocalsAccum[start + i] += recon.audio[i];
      count[start + i] += 1;
    }

    // Progress
    const frac = (ci + 1) / starts.length;
    const elapsed = (performance.now() - t0) / 1000;
    const eta = elapsed / frac * (1 - frac);
    
    onProgress?.({
      stage: 'processing',
      progress: Math.round(frac * 100),
      message: `Chunk ${ci + 1}/${starts.length} · ${elapsed.toFixed(1)}s elapsed · ~${eta.toFixed(0)}s remaining`,
      chunk: ci + 1,
      totalChunks: starts.length,
      elapsed,
      eta,
    });

    // Yield to UI
    await new Promise(r => setTimeout(r, 0));
  }

  // Average overlaps
  for (let i = 0; i < totalSamples; i++) {
    if (count[i] > 0) {
      vocalsAccum[i] /= count[i];
    }
  }

  onProgress?.({ stage: 'reconstructing', progress: 95, message: 'Building instrumental...' });

  // Build instrumental = original - vocals
  const instrAccum = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    instrAccum[i] = procMono[i] - vocalsAccum[i];
  }

  // Resample back to original sample rate if needed
  let finalVocals = vocalsAccum;
  let finalInstr = instrAccum;

  if (modelConfig.sampleRate !== originalSampleRate) {
    finalVocals = resample(vocalsAccum, modelConfig.sampleRate, originalSampleRate);
    finalInstr = resample(instrAccum, modelConfig.sampleRate, originalSampleRate);
  }

  // Trim to original length
  const origLen = audioBuffer.length;
  if (finalVocals.length > origLen) {
    finalVocals = finalVocals.slice(0, origLen);
    finalInstr = finalInstr.slice(0, origLen);
  }

  onProgress?.({ stage: 'reconstructing', progress: 98, message: 'Encoding WAV files...' });

  // Create AudioBuffers (mono)
  const vocalsCtx = new OfflineAudioContext(1, finalVocals.length, originalSampleRate);
  const vocalsBuf = vocalsCtx.createBuffer(1, finalVocals.length, originalSampleRate);
  const vocalsChannel = new Float32Array(finalVocals.length);
  vocalsChannel.set(finalVocals);
  vocalsBuf.copyToChannel(vocalsChannel, 0);

  const instrCtx = new OfflineAudioContext(1, finalInstr.length, originalSampleRate);
  const instrBuf = instrCtx.createBuffer(1, finalInstr.length, originalSampleRate);
  const instrChannel = new Float32Array(finalInstr.length);
  instrChannel.set(finalInstr);
  instrBuf.copyToChannel(instrChannel, 0);

  // Encode to WAV (mono)
  const vocalsWav = encodeMonoWAV(finalVocals, originalSampleRate);
  const instrWav = encodeMonoWAV(finalInstr, originalSampleRate);

  onProgress?.({ stage: 'done', progress: 100, message: 'Done!' });

  return [
    { stemName: 'Vocals', audioBuffer: vocalsBuf, wavData: vocalsWav },
    { stemName: 'Instrumental', audioBuffer: instrBuf, wavData: instrWav },
  ];
}

function encodeMonoWAV(audioData: Float32Array, sampleRate: number): ArrayBuffer {
  const nSamples = audioData.length;
  const buf = new ArrayBuffer(44 + nSamples * 2);
  const view = new DataView(buf);
  
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + nSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, nSamples * 2, true);
  
  let off = 44;
  for (let i = 0; i < nSamples; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    view.setInt16(off, sample * 32767, true);
    off += 2;
  }
  
  return buf;
}
