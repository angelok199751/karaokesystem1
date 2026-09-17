/**
 * Audio separation using HT-Demucs FT ONNX
 * Simple audio-to-audio separation - no manual STFT needed!
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
  const chunkSize = modelConfig.chunkSamples; // 343980
  
  // Calculate chunk positions with 25% overlap
  const overlap = 4; // 25% overlap
  const step = Math.floor(chunkSize / overlap);
  const starts: number[] = [];
  for (let s = 0; s < totalSamples; s += step) {
    starts.push(s);
  }

  // Accumulators for overlap-add
  // Output shape: (1, 4, 2, 343980) = [drums, bass, other, vocals]
  const stemsAccum: Float32Array[] = [
    new Float32Array(totalSamples * 2), // drums (stereo)
    new Float32Array(totalSamples * 2), // bass (stereo)
    new Float32Array(totalSamples * 2), // other (stereo)
    new Float32Array(totalSamples * 2), // vocals (stereo)
  ];
  const count = new Float32Array(totalSamples);

  const t0 = performance.now();

  for (let ci = 0; ci < starts.length; ci++) {
    const start = starts[ci];
    const end = Math.min(start + chunkSize, totalSamples);
    const chunkLen = end - start;

    // Extract chunk and pad to chunkSize
    const chunkL = new Float32Array(chunkSize);
    const chunkR = new Float32Array(chunkSize);
    chunkL.set(procLeft.subarray(start, end));
    chunkR.set(procRight.subarray(start, end));

    // Create input tensor: (1, 2, 343980) - interleaved stereo
    const inputData = new Float32Array(2 * chunkSize);
    for (let i = 0; i < chunkSize; i++) {
      inputData[i] = chunkL[i];           // Left channel
      inputData[chunkSize + i] = chunkR[i]; // Right channel
    }

    const inputTensor = new ort.Tensor('float32', inputData, [1, 2, chunkSize]);
    
    const feeds: Record<string, ort.Tensor> = {};
    feeds[modelConfig.inputName] = inputTensor;
    
    // Run ONNX inference
    const results = await session.run(feeds);
    const outputData = results[modelConfig.outputName].data as Float32Array;
    
    // Output shape: (1, 4, 2, chunkSize)
    // Extract each stem and accumulate
    for (let stemIdx = 0; stemIdx < 4; stemIdx++) {
      const stemOffset = stemIdx * 2 * chunkSize;
      
      for (let i = 0; i < chunkLen; i++) {
        // Left channel
        const leftVal = outputData[stemOffset + i];
        stemsAccum[stemIdx][start + i] += leftVal;
        
        // Right channel
        const rightVal = outputData[stemOffset + chunkSize + i];
        stemsAccum[stemIdx][totalSamples + start + i] += rightVal;
      }
    }
    
    for (let i = 0; i < chunkLen; i++) {
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

  onProgress?.({ stage: 'reconstructing', progress: 90, message: 'Averaging overlaps...' });

  // Average overlaps
  for (let stemIdx = 0; stemIdx < 4; stemIdx++) {
    for (let i = 0; i < totalSamples * 2; i++) {
      const sampleIdx = i % totalSamples;
      if (count[sampleIdx] > 0) {
        stemsAccum[stemIdx][i] /= count[sampleIdx];
      }
    }
  }

  onProgress?.({ stage: 'reconstructing', progress: 95, message: 'Building output...' });

  // Extract vocals (stem 3) and instrumental (drums + bass + other)
  const vocalsL = new Float32Array(totalSamples);
  const vocalsR = new Float32Array(totalSamples);
  const instrL = new Float32Array(totalSamples);
  const instrR = new Float32Array(totalSamples);

  for (let i = 0; i < totalSamples; i++) {
    // Vocals = stem 3
    vocalsL[i] = stemsAccum[3][i];
    vocalsR[i] = stemsAccum[3][totalSamples + i];
    
    // Instrumental = drums (0) + bass (1) + other (2)
    instrL[i] = stemsAccum[0][i] + stemsAccum[1][i] + stemsAccum[2][i];
    instrR[i] = stemsAccum[0][totalSamples + i] + stemsAccum[1][totalSamples + i] + stemsAccum[2][totalSamples + i];
  }

  // Normalize
  function normalize(signal: Float32Array, targetPeak: number = 0.95): Float32Array {
    let maxVal = 0;
    for (let i = 0; i < signal.length; i++) {
      maxVal = Math.max(maxVal, Math.abs(signal[i]));
    }
    if (maxVal < 1e-8) return signal;
    const gain = targetPeak / maxVal;
    const result = new Float32Array(signal.length);
    for (let i = 0; i < signal.length; i++) {
      result[i] = signal[i] * gain;
    }
    return result;
  }

  const vocalsLNorm = normalize(vocalsL);
  const vocalsRNorm = normalize(vocalsR);
  const instrLNorm = normalize(instrL);
  const instrRNorm = normalize(instrR);

  onProgress?.({ stage: 'reconstructing', progress: 98, message: 'Resampling and encoding...' });

  // Resample back to original sample rate if needed
  let finalVocalsL: Float32Array = vocalsLNorm;
  let finalVocalsR: Float32Array = vocalsRNorm;
  let finalInstrL: Float32Array = instrLNorm;
  let finalInstrR: Float32Array = instrRNorm;

  if (modelConfig.sampleRate !== originalSampleRate) {
    finalVocalsL = resample(vocalsLNorm, modelConfig.sampleRate, originalSampleRate);
    finalVocalsR = resample(vocalsRNorm, modelConfig.sampleRate, originalSampleRate);
    finalInstrL = resample(instrLNorm, modelConfig.sampleRate, originalSampleRate);
    finalInstrR = resample(instrRNorm, modelConfig.sampleRate, originalSampleRate);
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
