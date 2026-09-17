/**
 * Audio separation using demucs-web
 * Optimized for browser with 170MB model
 */
import { DemucsProcessor } from 'demucs-web';
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

  onProgress?.({ stage: 'processing', progress: 10, message: 'Initializing Demucs processor...' });

  // Create DemucsProcessor
  const processor = new DemucsProcessor({
    ort,
    onProgress: (p: any) => {
      const progress = typeof p === 'number' ? p : (p.progress || 0);
      onProgress?.({
        stage: 'processing',
        progress: 20 + progress * 70,
        message: `Processing: ${Math.round(progress * 100)}%`,
      });
    },
    onLog: (phase: string, msg: string) => {
      console.log(`[Demucs:${phase}] ${msg}`);
    },
  });

  // Load model into processor
  await processor.loadModel(modelConfig.url);

  onProgress?.({ stage: 'processing', progress: 30, message: 'Separating audio...' });

  // Separate audio
  const result = await processor.separate(procLeft, procRight);

  onProgress?.({ stage: 'reconstructing', progress: 90, message: 'Building output...' });

  // Extract vocals and instrumental
  const vocalsL = result.vocals.left;
  const vocalsR = result.vocals.right;
  
  // Instrumental = drums + bass + other
  const instrL = new Float32Array(vocalsL.length);
  const instrR = new Float32Array(vocalsR.length);
  for (let i = 0; i < vocalsL.length; i++) {
    instrL[i] = result.drums.left[i] + result.bass.left[i] + result.other.left[i];
    instrR[i] = result.drums.right[i] + result.bass.right[i] + result.other.right[i];
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

  onProgress?.({ stage: 'reconstructing', progress: 95, message: 'Resampling and encoding...' });

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
