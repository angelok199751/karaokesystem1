/**
 * Audio separation using UVR-MDX-NET ONNX models
 * Lightweight vocal separation (28-64 MB)
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

/** Real FFT with center padding */
function stftCenter(signal: Float32Array, nFft: number, hop: number, win: Float32Array): { real: Float32Array[]; imag: Float32Array[]; numFrames: number } {
  const nFreq = nFft / 2 + 1;
  const padLength = nFft / 2;
  
  // Pad signal with reflection (center=True)
  const padded = new Float32Array(signal.length + 2 * padLength);
  for (let i = 0; i < padLength; i++) {
    padded[padLength - 1 - i] = signal[i];
    padded[padLength + signal.length + i] = signal[signal.length - 1 - i];
  }
  padded.set(signal, padLength);
  
  const numFrames = Math.floor((padded.length - nFft) / hop) + 1;
  const real: Float32Array[] = [];
  const imag: Float32Array[] = [];
  
  for (let t = 0; t < numFrames; t++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    const off = t * hop;
    
    for (let i = 0; i < nFft; i++) {
      frameRe[i] = padded[off + i] * win[i];
    }
    
    fftInPlace(frameRe, frameIm, nFft);
    
    const posRe = new Float32Array(nFreq);
    const posIm = new Float32Array(nFreq);
    for (let f = 0; f < nFreq; f++) {
      posRe[f] = frameRe[f];
      posIm[f] = frameIm[f];
    }
    
    real.push(posRe);
    imag.push(posIm);
  }
  
  return { real, imag, numFrames };
}

/** Inverse STFT with center padding removal */
function istftCenter(real: Float32Array[], imag: Float32Array[], nFft: number, hop: number, win: Float32Array, originalLength: number): Float32Array {
  const numFrames = real.length;
  const padLength = nFft / 2;
  const paddedLength = originalLength + 2 * padLength;
  
  const output = new Float32Array(paddedLength);
  const winSum = new Float32Array(paddedLength);
  
  for (let t = 0; t < numFrames; t++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    const nFreq = real[t].length;
    
    for (let f = 0; f < nFreq; f++) {
      frameRe[f] = real[t][f];
      frameIm[f] = imag[t][f];
    }
    for (let f = 1; f < nFreq; f++) {
      frameRe[nFft - f] = real[t][f];
      frameIm[nFft - f] = -imag[t][f];
    }
    
    fftInPlace(frameRe, frameIm, nFft);
    
    const off = t * hop;
    for (let i = 0; i < nFft && off + i < paddedLength; i++) {
      output[off + i] += frameRe[i] * win[i];
      winSum[off + i] += win[i] * win[i];
    }
  }
  
  // Normalize
  for (let i = 0; i < paddedLength; i++) {
    if (winSum[i] > 1e-8) {
      output[i] /= winSum[i];
    }
  }
  
  // Remove padding
  return output.slice(padLength, padLength + originalLength);
}

// ── Prepare model input ────────────────────────────────────────────────────

interface ChunkInput {
  magnitude: Float32Array;  // [1, 1, n_freq, n_frames]
  numFrames: number;
  stftL: { real: Float32Array[]; imag: Float32Array[] };
  stftR: { real: Float32Array[]; imag: Float32Array[] };
}

function prepareChunkInput(
  left: Float32Array,
  right: Float32Array,
  win: Float32Array,
  nFft: number,
  hop: number
): ChunkInput {
  const nFreq = nFft / 2 + 1;
  
  // Mix to mono for magnitude computation
  const mono = new Float32Array(left.length);
  for (let i = 0; i < left.length; i++) {
    mono[i] = (left[i] + right[i]) / 2;
  }
  
  // STFT for mono
  const stftMono = stftCenter(mono, nFft, hop, win);
  
  // Also compute STFT for stereo (for later application)
  const stftL = stftCenter(left, nFft, hop, win);
  const stftR = stftCenter(right, nFft, hop, win);
  
  const numFrames = stftMono.numFrames;
  
  // Compute magnitude [1, 1, n_freq, n_frames]
  const magnitude = new Float32Array(nFreq * numFrames);
  
  for (let t = 0; t < numFrames; t++) {
    for (let f = 0; f < nFreq; f++) {
      const re = stftMono.real[t][f];
      const im = stftMono.imag[t][f];
      magnitude[f * numFrames + t] = Math.sqrt(re * re + im * im);
    }
  }
  
  return { magnitude, numFrames, stftL, stftR };
}

// ── Apply model output and iSTFT ───────────────────────────────────────────

interface StemAudio {
  left: Float32Array;
  right: Float32Array;
}

function applyMaskAndReconstruct(
  mask: Float32Array,
  stftL: { real: Float32Array[]; imag: Float32Array[] },
  stftR: { real: Float32Array[]; imag: Float32Array[] },
  numFrames: number,
  win: Float32Array,
  nFft: number,
  hop: number,
  originalLength: number
): StemAudio {
  const nFreq = nFft / 2 + 1;
  
  // Apply mask to stereo STFT
  const maskedRealL: Float32Array[] = [];
  const maskedImagL: Float32Array[] = [];
  const maskedRealR: Float32Array[] = [];
  const maskedImagR: Float32Array[] = [];
  
  for (let t = 0; t < numFrames; t++) {
    const mReL = new Float32Array(nFreq);
    const mImL = new Float32Array(nFreq);
    const mReR = new Float32Array(nFreq);
    const mImR = new Float32Array(nFreq);
    
    for (let f = 0; f < nFreq; f++) {
      const m = mask[f * numFrames + t];
      
      mReL[f] = stftL.real[t][f] * m;
      mImL[f] = stftL.imag[t][f] * m;
      mReR[f] = stftR.real[t][f] * m;
      mImR[f] = stftR.imag[t][f] * m;
    }
    
    maskedRealL.push(mReL);
    maskedImagL.push(mImL);
    maskedRealR.push(mReR);
    maskedImagR.push(mImR);
  }
  
  // iSTFT for both channels
  const left = istftCenter(maskedRealL, maskedImagL, nFft, hop, win, originalLength);
  const right = istftCenter(maskedRealR, maskedImagR, nFft, hop, win, originalLength);
  
  return { left, right };
}

// ── Resample ────────────────────────────────────────────────────────────────

function resample(audioData: Float32Array, fromRate: number, toRate: number): Float32Array {
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
  
  // Calculate chunk positions with overlap
  const step = Math.floor(modelConfig.chunkSamples / modelConfig.overlap);
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
    const end = Math.min(start + modelConfig.chunkSamples, totalSamples);
    const chunkLen = end - start;

    // Extract & pad chunk
    const cL = new Float32Array(modelConfig.chunkSamples);
    const cR = new Float32Array(modelConfig.chunkSamples);
    cL.set(procLeft.subarray(start, end));
    cR.set(procRight.subarray(start, end));

    // STFT & prepare input
    const { magnitude, numFrames, stftL, stftR } = prepareChunkInput(
      cL, cR, win, modelConfig.nFft, modelConfig.hopLength
    );

    // Run ONNX inference
    const nFreq = modelConfig.nFft / 2 + 1;
    const inputTensor = new ort.Tensor('float32', magnitude, [1, 1, nFreq, numFrames]);
    
    const feeds: Record<string, ort.Tensor> = {};
    feeds[modelConfig.inputName] = inputTensor;
    
    const results = await session.run(feeds);
    const mask = results[modelConfig.outputName].data as Float32Array;

    // Apply mask and reconstruct vocals
    const vocals = applyMaskAndReconstruct(
      mask, stftL, stftR, numFrames, win,
      modelConfig.nFft, modelConfig.hopLength, modelConfig.chunkSamples
    );

    // Accumulate with overlap
    for (let i = 0; i < chunkLen; i++) {
      vocalsL[start + i] += vocals.left[i];
      vocalsR[start + i] += vocals.right[i];
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

  onProgress?.({ stage: 'reconstructing', progress: 90, message: 'Building instrumental...' });

  // Build instrumental = original - vocals
  const instrL = new Float32Array(totalSamples);
  const instrR = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    instrL[i] = procLeft[i] - vocalsL[i];
    instrR[i] = procRight[i] - vocalsR[i];
  }

  onProgress?.({ stage: 'reconstructing', progress: 95, message: 'Resampling and encoding...' });

  // Resample back to original sample rate if needed
  let finalVocalsL: Float32Array = vocalsL;
  let finalVocalsR: Float32Array = vocalsR;
  let finalInstrL: Float32Array = instrL;
  let finalInstrR: Float32Array = instrR;

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

  // Create AudioBuffers
  const vocalsCtx = new OfflineAudioContext(2, finalVocalsL.length, originalSampleRate);
  const vocalsBuf = vocalsCtx.createBuffer(2, finalVocalsL.length, originalSampleRate);
  const vocalsLData = new Float32Array(finalVocalsL.length);
  vocalsLData.set(finalVocalsL);
  const vocalsRData = new Float32Array(finalVocalsR.length);
  vocalsRData.set(finalVocalsR);
  vocalsBuf.copyToChannel(vocalsLData, 0);
  vocalsBuf.copyToChannel(vocalsRData, 1);

  const instrCtx = new OfflineAudioContext(2, finalInstrL.length, originalSampleRate);
  const instrBuf = instrCtx.createBuffer(2, finalInstrL.length, originalSampleRate);
  const instrLData = new Float32Array(finalInstrL.length);
  instrLData.set(finalInstrL);
  const instrRData = new Float32Array(finalInstrR.length);
  instrRData.set(finalInstrR);
  instrBuf.copyToChannel(instrLData, 0);
  instrBuf.copyToChannel(instrRData, 1);

  // Encode to WAV
  const vocalsStereo = interleaveStereo(finalVocalsL, finalVocalsR);
  const instrStereo = interleaveStereo(finalInstrL, finalInstrR);
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
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
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
