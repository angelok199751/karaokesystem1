/**
 * Audio separation using UVR-MDX-NET ONNX models
 * Correct format: [batch, 4, dim_f, dim_t] where 4 = stereo(2) × complex(2)
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

/**
 * STFT for MDX-Net format
 * Input: stereo audio [2, chunk_size]
 * Output: [1, 4, dim_f, dim_t] where 4 = stereo(2) × complex(2)
 */
function stftMDX(
  left: Float32Array,
  right: Float32Array,
  nFft: number,
  hop: number,
  dimF: number,
  dimT: number
): Float32Array {
  const window = hannWindow(nFft);
  const nBins = nFft / 2 + 1;
  const numFrames = Math.pow(2, dimT); // 2^8 = 256
  
  // Output: [1, 4, dim_f, dim_t]
  // Channels: [L_real, L_imag, R_real, R_imag]
  const output = new Float32Array(4 * dimF * numFrames);
  
  // Process left channel
  for (let t = 0; t < numFrames; t++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    const off = t * hop;
    
    for (let i = 0; i < nFft && off + i < left.length; i++) {
      frameRe[i] = left[off + i] * window[i];
    }
    
    fftInPlace(frameRe, frameIm, nFft);
    
    // Store first dimF bins (truncate from nBins to dimF)
    for (let f = 0; f < dimF && f < nBins; f++) {
      // Channel 0: L_real
      output[0 * dimF * numFrames + f * numFrames + t] = frameRe[f];
      // Channel 1: L_imag
      output[1 * dimF * numFrames + f * numFrames + t] = frameIm[f];
    }
  }
  
  // Process right channel
  for (let t = 0; t < numFrames; t++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    const off = t * hop;
    
    for (let i = 0; i < nFft && off + i < right.length; i++) {
      frameRe[i] = right[off + i] * window[i];
    }
    
    fftInPlace(frameRe, frameIm, nFft);
    
    for (let f = 0; f < dimF && f < nBins; f++) {
      // Channel 2: R_real
      output[2 * dimF * numFrames + f * numFrames + t] = frameRe[f];
      // Channel 3: R_imag
      output[3 * dimF * numFrames + f * numFrames + t] = frameIm[f];
    }
  }
  
  return output;
}

/**
 * iSTFT for MDX-Net format
 * Input: [1, 4, dim_f, dim_t] masked spectrogram
 * Output: stereo audio [2, length]
 */
function istftMDX(
  spec: Float32Array,
  nFft: number,
  hop: number,
  dimF: number,
  dimT: number,
  outputLength: number
): { left: Float32Array; right: Float32Array } {
  const window = hannWindow(nFft);
  const nBins = nFft / 2 + 1;
  const numFrames = Math.pow(2, dimT); // 2^8 = 256
  
  const left = new Float32Array(outputLength);
  const right = new Float32Array(outputLength);
  const winSum = new Float32Array(outputLength);
  
  // Process left channel (channels 0, 1)
  for (let t = 0; t < numFrames; t++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    
    // Copy dimF bins
    for (let f = 0; f < dimF && f < nBins; f++) {
      frameRe[f] = spec[0 * dimF * numFrames + f * numFrames + t];
      frameIm[f] = spec[1 * dimF * numFrames + f * numFrames + t];
    }
    
    // Mirror for negative frequencies
    for (let f = 1; f < nBins; f++) {
      if (f >= dimF) break;
      frameRe[nFft - f] = frameRe[f];
      frameIm[nFft - f] = -frameIm[f];
    }
    
    fftInPlace(frameRe, frameIm, nFft);
    
    const off = t * hop;
    for (let i = 0; i < nFft && off + i < outputLength; i++) {
      left[off + i] += frameRe[i] * window[i];
      winSum[off + i] += window[i] * window[i];
    }
  }
  
  // Process right channel (channels 2, 3)
  const winSumR = new Float32Array(outputLength);
  for (let t = 0; t < numFrames; t++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    
    for (let f = 0; f < dimF && f < nBins; f++) {
      frameRe[f] = spec[2 * dimF * numFrames + f * numFrames + t];
      frameIm[f] = spec[3 * dimF * numFrames + f * numFrames + t];
    }
    
    for (let f = 1; f < nBins; f++) {
      if (f >= dimF) break;
      frameRe[nFft - f] = frameRe[f];
      frameIm[nFft - f] = -frameIm[f];
    }
    
    fftInPlace(frameRe, frameIm, nFft);
    
    const off = t * hop;
    for (let i = 0; i < nFft && off + i < outputLength; i++) {
      right[off + i] += frameRe[i] * window[i];
      winSumR[off + i] += window[i] * window[i];
    }
  }
  
  // Normalize
  for (let i = 0; i < outputLength; i++) {
    if (winSum[i] > 1e-8) left[i] /= winSum[i];
    if (winSumR[i] > 1e-8) right[i] /= winSumR[i];
  }
  
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
  
  // Calculate chunk positions with overlap
  const chunkSize = modelConfig.chunkSamples;
  const step = Math.floor(chunkSize / modelConfig.overlap);
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
    const end = Math.min(start + chunkSize, totalSamples);
    const chunkLen = end - start;

    // Extract chunk
    const cL = new Float32Array(chunkSize);
    const cR = new Float32Array(chunkSize);
    cL.set(procLeft.subarray(start, end));
    cR.set(procRight.subarray(start, end));

    // Pad to chunk_size if needed
    const paddedL = new Float32Array(chunkSize);
    const paddedR = new Float32Array(chunkSize);
    paddedL.set(cL);
    paddedR.set(cR);

    // STFT: [1, 4, dim_f, dim_t]
    const spek = stftMDX(
      paddedL,
      paddedR,
      modelConfig.nFft,
      modelConfig.hopLength,
      modelConfig.dimF,
      modelConfig.dimT
    );

    // Run ONNX inference
    const numFrames = Math.pow(2, modelConfig.dimT); // 2^8 = 256
    const inputTensor = new ort.Tensor('float32', spek, [1, 4, modelConfig.dimF, numFrames]);
    
    const feeds: Record<string, ort.Tensor> = {};
    feeds[modelConfig.inputName] = inputTensor;
    
    const results = await session.run(feeds);
    const outputSpec = results[modelConfig.outputName].data as Float32Array;

    // iSTFT: get vocals
    const vocals = istftMDX(
      outputSpec,
      modelConfig.nFft,
      modelConfig.hopLength,
      modelConfig.dimF,
      modelConfig.dimT,
      chunkSize
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
