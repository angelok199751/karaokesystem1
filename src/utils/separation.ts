/**
 * Audio separation using BS PolarFormer ONNX model
 * Based on the reference implementation from bgkb/bs_polarformer
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

// ── Prepare model input from stereo chunk ───────────────────────────────────

interface ChunkInput {
  input: Float32Array;
  nFrames: number;
  stftL: STFTResult;
  stftR: STFTResult;
}

function prepareChunkInput(
  left: Float32Array,
  right: Float32Array,
  win: Float32Array,
  nFft: number,
  hop: number,
  nFreq: number
): ChunkInput {
  const stftL = stft(left, nFft, hop, win, nFreq);
  const stftR = stft(right, nFft, hop, win, nFreq);
  const nFrames = stftL.nFrames;
  
  // Interleave: (f_left, f_right) for each freq
  // Shape: (nFrames, n_freq*2*2) = (nFrames, 4100)
  const totalFreqs = nFreq * 2; // 2050
  const featDim = totalFreqs * 2; // 4100
  const input = new Float32Array(nFrames * featDim);
  
  for (let t = 0; t < nFrames; t++) {
    for (let f = 0; f < nFreq; f++) {
      const lRe = stftL.data[(f * nFrames + t) * 2];
      const lIm = stftL.data[(f * nFrames + t) * 2 + 1];
      const rRe = stftR.data[(f * nFrames + t) * 2];
      const rIm = stftR.data[(f * nFrames + t) * 2 + 1];
      
      // band order: (f*2) = left, (f*2+1) = right
      const base = t * featDim;
      input[base + (f * 2) * 2] = lRe;
      input[base + (f * 2) * 2 + 1] = lIm;
      input[base + (f * 2 + 1) * 2] = rRe;
      input[base + (f * 2 + 1) * 2 + 1] = rIm;
    }
  }
  
  return { input, nFrames, stftL, stftR };
}

// ── Apply mask and iSTFT ────────────────────────────────────────────────────

interface ReconstructedAudio {
  left: Float32Array;
  right: Float32Array;
}

function applyMaskAndReconstruct(
  mask: Float32Array,
  stftL: STFTResult,
  stftR: STFTResult,
  nFrames: number,
  win: Float32Array,
  nFft: number,
  hop: number,
  nFreq: number,
  length: number
): ReconstructedAudio {
  // mask shape: [1, 1, 2050, nFrames, 2] flattened
  const maskedL = new Float32Array(nFreq * nFrames * 2);
  const maskedR = new Float32Array(nFreq * nFrames * 2);

  for (let f = 0; f < nFreq; f++) {
    for (let t = 0; t < nFrames; t++) {
      // mask indices
      const mLIdx = ((f * 2) * nFrames + t) * 2;
      const mRIdx = ((f * 2 + 1) * nFrames + t) * 2;
      const mLRe = mask[mLIdx], mLIm = mask[mLIdx + 1];
      const mRRe = mask[mRIdx], mRIm = mask[mRIdx + 1];

      const sIdx = (f * nFrames + t) * 2;
      const sLRe = stftL.data[sIdx], sLIm = stftL.data[sIdx + 1];
      const sRRe = stftR.data[sIdx], sRIm = stftR.data[sIdx + 1];

      // Complex multiply
      maskedL[sIdx] = sLRe * mLRe - sLIm * mLIm;
      maskedL[sIdx + 1] = sLRe * mLIm + sLIm * mLRe;
      maskedR[sIdx] = sRRe * mRRe - sRIm * mRIm;
      maskedR[sIdx + 1] = sRRe * mRIm + sRIm * mRRe;
    }
  }
  
  // Zero DC bin
  for (let t = 0; t < nFrames; t++) {
    maskedL[t * 2] = 0; maskedL[t * 2 + 1] = 0;
    maskedR[t * 2] = 0; maskedR[t * 2 + 1] = 0;
  }

  const reconL = istft(maskedL, nFrames, nFft, hop, win, nFreq, length);
  const reconR = istft(maskedR, nFrames, nFft, hop, win, nFreq, length);
  return { left: reconL, right: reconR };
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

  // Get stereo channels
  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.numberOfChannels > 1 
    ? audioBuffer.getChannelData(1) 
    : audioBuffer.getChannelData(0);

  // Resample to model's sample rate if needed
  const procLeft = modelConfig.sampleRate !== originalSampleRate
    ? resample(left, originalSampleRate, modelConfig.sampleRate)
    : new Float32Array(left);
  const procRight = modelConfig.sampleRate !== originalSampleRate
    ? resample(right, originalSampleRate, modelConfig.sampleRate)
    : new Float32Array(right);

  const totalSamples = procLeft.length;
  const win = hannWindow(modelConfig.winLength);
  const step = Math.floor(modelConfig.chunkSize / modelConfig.overlap);
  
  // Calculate chunk positions
  const starts: number[] = [];
  for (let s = 0; s < totalSamples; s += step) {
    starts.push(s);
  }

  // Accumulators for overlap-add
  const vocalsL = new Float32Array(totalSamples);
  const vocalsR = new Float32Array(totalSamples);
  const count = new Float32Array(totalSamples);

  const t0 = performance.now();

  for (let ci = 0; ci < starts.length; ci++) {
    const start = starts[ci];
    const end = Math.min(start + modelConfig.chunkSize, totalSamples);
    const chunkLen = end - start;

    // Extract & pad chunk
    const cL = new Float32Array(modelConfig.chunkSize);
    const cR = new Float32Array(modelConfig.chunkSize);
    cL.set(procLeft.subarray(start, end));
    cR.set(procRight.subarray(start, end));

    // STFT & prepare input
    const { input, nFrames, stftL, stftR } = prepareChunkInput(
      cL, cR, win, modelConfig.nFft, modelConfig.hopLength, nFreq
    );

    // Run ONNX inference
    const tensor = new ort.Tensor('float32', input, [1, nFrames, nFreq * 2 * 2]);
    const results = await session.run({ [modelConfig.inputName]: tensor });
    const mask = results[modelConfig.outputName].data as Float32Array;

    // Reconstruct
    const recon = applyMaskAndReconstruct(
      mask, stftL, stftR, nFrames, win,
      modelConfig.nFft, modelConfig.hopLength, nFreq, modelConfig.chunkSize
    );

    // Accumulate with overlap
    for (let i = 0; i < chunkLen; i++) {
      vocalsL[start + i] += recon.left[i];
      vocalsR[start + i] += recon.right[i];
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
      vocalsL[i] /= count[i];
      vocalsR[i] /= count[i];
    }
  }

  onProgress?.({ stage: 'reconstructing', progress: 95, message: 'Building instrumental...' });

  // Build instrumental = original - vocals
  const instrL = new Float32Array(totalSamples);
  const instrR = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    instrL[i] = procLeft[i] - vocalsL[i];
    instrR[i] = procRight[i] - vocalsR[i];
  }

  // Resample back to original sample rate if needed
  let finalVocalsL = vocalsL;
  let finalVocalsR = vocalsR;
  let finalInstrL = instrL;
  let finalInstrR = instrR;

  if (modelConfig.sampleRate !== originalSampleRate) {
    finalVocalsL = resample(vocalsL, modelConfig.sampleRate, originalSampleRate);
    finalVocalsR = resample(vocalsR, modelConfig.sampleRate, originalSampleRate);
    finalInstrL = resample(instrL, modelConfig.sampleRate, originalSampleRate);
    finalInstrR = resample(instrR, modelConfig.sampleRate, originalSampleRate);
  }

  // Trim to original length
  const origLen = audioBuffer.length;
  if (finalVocalsL.length > origLen) {
    finalVocalsL = finalVocalsL.slice(0, origLen);
    finalVocalsR = finalVocalsR.slice(0, origLen);
    finalInstrL = finalInstrL.slice(0, origLen);
    finalInstrR = finalInstrR.slice(0, origLen);
  }

  onProgress?.({ stage: 'reconstructing', progress: 98, message: 'Encoding WAV files...' });

  // Create stereo interleaved data for WAV encoding
  const vocalsStereo = interleaveStereo(finalVocalsL, finalVocalsR);
  const instrStereo = interleaveStereo(finalInstrL, finalInstrR);

  // Create AudioBuffers
  const vocalsCtx = new OfflineAudioContext(2, finalVocalsL.length, originalSampleRate);
  const vocalsBuf = vocalsCtx.createBuffer(2, finalVocalsL.length, originalSampleRate);
  vocalsBuf.copyToChannel(new Float32Array(finalVocalsL), 0);
  vocalsBuf.copyToChannel(new Float32Array(finalVocalsR), 1);

  const instrCtx = new OfflineAudioContext(2, finalInstrL.length, originalSampleRate);
  const instrBuf = instrCtx.createBuffer(2, finalInstrL.length, originalSampleRate);
  instrBuf.copyToChannel(new Float32Array(finalInstrL), 0);
  instrBuf.copyToChannel(new Float32Array(finalInstrR), 1);

  // Encode to WAV (stereo)
  const vocalsWav = encodeStereoWAV(vocalsStereo, originalSampleRate);
  const instrWav = encodeStereoWAV(instrStereo, originalSampleRate);

  onProgress?.({ stage: 'done', progress: 100, message: 'Done!' });

  return [
    { stemName: 'Vocals', audioBuffer: vocalsBuf, wavData: vocalsWav },
    { stemName: 'Instrumental', audioBuffer: instrBuf, wavData: instrWav },
  ];
}

function interleaveStereo(left: Float32Array, right: Float32Array): Float32Array {
  const out = new Float32Array(left.length * 2);
  for (let i = 0; i < left.length; i++) {
    out[i * 2] = left[i];
    out[i * 2 + 1] = right[i];
  }
  return out;
}

function encodeStereoWAV(interleaved: Float32Array, sampleRate: number): ArrayBuffer {
  const nSamples = interleaved.length / 2;
  const buf = new ArrayBuffer(44 + interleaved.length * 2);
  const view = new DataView(buf);
  
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + interleaved.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 2, true); // stereo
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true); // byte rate
  view.setUint16(32, 4, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, interleaved.length * 2, true);
  
  let off = 44;
  for (let i = 0; i < interleaved.length; i++) {
    const sample = Math.max(-1, Math.min(1, interleaved[i]));
    view.setInt16(off, sample * 32767, true);
    off += 2;
  }
  
  return buf;
}
