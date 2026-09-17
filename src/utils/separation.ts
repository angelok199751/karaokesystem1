/**
 * Audio separation logic using ONNX models
 * Implements spectral masking approach for source separation
 */
import * as ort from 'onnxruntime-web';
import { ModelConfig } from './modelManager';
import {
  audioBufferToMono,
  computeSTFT,
  computeISTFT,
  computeMagnitude,
  encodeWAV,
  STFTParams,
} from './audioProcessor';

export interface SeparationResult {
  stemName: string;
  audioBuffer: AudioBuffer;
  wavData: ArrayBuffer;
}

export interface SeparationProgress {
  stage: 'loading' | 'preparing' | 'processing' | 'reconstructing' | 'done';
  progress: number; // 0-100
  message: string;
}

/**
 * Resample audio to target sample rate using linear interpolation
 */
function resample(audioData: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return new Float32Array(audioData);
  
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

/**
 * Pad signal to be divisible by hop length
 */
function padSignal(signal: Float32Array, hopLength: number): Float32Array {
  const padLength = hopLength - (signal.length % hopLength);
  if (padLength === hopLength) return signal;
  
  const padded = new Float32Array(signal.length + padLength);
  padded.set(signal);
  return padded;
}

/**
 * Process audio through model for vocal separation using spectral masking
 */
export async function separateAudio(
  audioBuffer: AudioBuffer,
  modelConfig: ModelConfig,
  session: ort.InferenceSession,
  onProgress?: (progress: SeparationProgress) => void
): Promise<SeparationResult[]> {
  const sampleRate = audioBuffer.sampleRate;
  
  onProgress?.({ stage: 'preparing', progress: 0, message: 'Preparing audio...' });

  // Convert to mono
  const monoSignal = audioBufferToMono(audioBuffer);
  
  // Resample if needed
  const signal = modelConfig.sampleRate !== sampleRate
    ? resample(monoSignal, sampleRate, modelConfig.sampleRate)
    : monoSignal;

  // STFT parameters
  const stftParams: STFTParams = {
    nFft: modelConfig.nFft,
    hopLength: modelConfig.hopLength,
    winLength: modelConfig.nFft,
  };

  // Process in chunks
  const chunkSamples = modelConfig.chunkDuration * modelConfig.sampleRate;
  const numChunks = Math.ceil(signal.length / chunkSamples);
  
  // Accumulate full-length outputs
  const stemSignals: Float32Array[] = modelConfig.stems.map(() => new Float32Array(signal.length));

  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    const start = chunkIdx * chunkSamples;
    const end = Math.min(start + chunkSamples, signal.length);
    const chunk = signal.slice(start, end);
    
    const paddedChunk = padSignal(chunk, modelConfig.hopLength);
    
    onProgress?.({
      stage: 'processing',
      progress: Math.round((chunkIdx / numChunks) * 80) + 10,
      message: `Processing chunk ${chunkIdx + 1}/${numChunks}...`,
    });

    // Compute STFT for this chunk
    const { real, imag } = computeSTFT(paddedChunk, stftParams);
    const magnitude = computeMagnitude(real, imag);

    // Prepare input tensor
    const numFrames = magnitude.length;
    const numFreqs = magnitude[0].length;
    
    // Create flat input array [1, 1, numFreqs, numFrames]
    const inputData = new Float32Array(numFreqs * numFrames);
    for (let t = 0; t < numFrames; t++) {
      for (let f = 0; f < numFreqs; f++) {
        inputData[t * numFreqs + f] = magnitude[t][f];
      }
    }

    // Get input names from session
    const inputNames = session.inputNames;
    const inputName = inputNames[0];
    
    // Determine input shape
    const inputShape = [1, 1, numFreqs, numFrames];

    // Create tensor
    const inputTensor = new ort.Tensor('float32', inputData, inputShape);

    try {
      // Run inference
      const feeds: Record<string, ort.Tensor> = {};
      feeds[inputName] = inputTensor;
      
      const output = await session.run(feeds);
      const outputNames = session.outputNames;
      
      const outputTensor = output[outputNames[0]];
      const outputData = outputTensor.data as Float32Array;

      // Process output
      let masks: Float32Array[];
      
      if (modelConfig.stems.length === 2) {
        const vocalMask = extractMask(outputData, numFreqs, numFrames);
        const instrumentalMask = new Float32Array(vocalMask.length);
        for (let i = 0; i < vocalMask.length; i++) {
          instrumentalMask[i] = 1 - vocalMask[i];
        }
        masks = [vocalMask, instrumentalMask];
      } else {
        masks = extractMultiMask(outputData, modelConfig.stems.length, numFreqs, numFrames);
      }

      // Apply masks and reconstruct
      for (let stemIdx = 0; stemIdx < modelConfig.stems.length; stemIdx++) {
        const mask2D = reshapeMask(masks[stemIdx], numFrames, numFreqs);
        
        const maskedReal: Float32Array[] = [];
        const maskedImag: Float32Array[] = [];
        
        for (let t = 0; t < numFrames; t++) {
          const mRe = new Float32Array(numFreqs);
          const mIm = new Float32Array(numFreqs);
          for (let f = 0; f < numFreqs; f++) {
            const m = Math.max(0, Math.min(1, mask2D[t][f]));
            mRe[f] = real[t][f] * m;
            mIm[f] = imag[t][f] * m;
          }
          maskedReal.push(mRe);
          maskedImag.push(mIm);
        }

        const reconstructed = computeISTFT(maskedReal, maskedImag, stftParams, paddedChunk.length);
        const validLength = Math.min(reconstructed.length, chunk.length);
        for (let i = 0; i < validLength; i++) {
          stemSignals[stemIdx][start + i] = reconstructed[i];
        }
      }
    } catch (err) {
      console.warn(`Model inference failed for chunk ${chunkIdx}, using fallback:`, err);
      
      // Fallback: frequency-band based separation
      const fallbackMasks = createFallbackMasks(numFrames, numFreqs, modelConfig.stems.length);
      
      for (let stemIdx = 0; stemIdx < modelConfig.stems.length; stemIdx++) {
        const mask2D = reshapeMask(fallbackMasks[stemIdx], numFrames, numFreqs);
        const maskedReal: Float32Array[] = [];
        const maskedImag: Float32Array[] = [];
        
        for (let t = 0; t < numFrames; t++) {
          const mRe = new Float32Array(numFreqs);
          const mIm = new Float32Array(numFreqs);
          for (let f = 0; f < numFreqs; f++) {
            mRe[f] = real[t][f] * mask2D[t][f];
            mIm[f] = imag[t][f] * mask2D[t][f];
          }
          maskedReal.push(mRe);
          maskedImag.push(mIm);
        }

        const reconstructed = computeISTFT(maskedReal, maskedImag, stftParams, paddedChunk.length);
        const validLength = Math.min(reconstructed.length, chunk.length);
        for (let i = 0; i < validLength; i++) {
          stemSignals[stemIdx][start + i] = reconstructed[i];
        }
      }
    }
  }

  onProgress?.({ stage: 'reconstructing', progress: 90, message: 'Reconstructing audio...' });

  // Create AudioBuffers and WAV data for each stem
  const finalResults: SeparationResult[] = [];
  
  for (let stemIdx = 0; stemIdx < modelConfig.stems.length; stemIdx++) {
    let stemSignal = stemSignals[stemIdx];
    
    // Resample back to original sample rate if needed
    if (modelConfig.sampleRate !== sampleRate) {
      stemSignal = resample(stemSignal, modelConfig.sampleRate, sampleRate);
    }
    
    // Trim to original length
    const originalLength = audioBuffer.length;
    if (stemSignal.length > originalLength) {
      stemSignal = stemSignal.slice(0, originalLength);
    }
    
    // Create AudioBuffer (mono)
    const ctx = new OfflineAudioContext(1, stemSignal.length, sampleRate);
    const buf = ctx.createBuffer(1, stemSignal.length, sampleRate);
    // Copy data into a new Float32Array to avoid type issues
    const channelData = new Float32Array(stemSignal.length);
    for (let i = 0; i < stemSignal.length; i++) {
      channelData[i] = stemSignal[i];
    }
    buf.copyToChannel(channelData, 0);
    
    // Encode to WAV
    const wavData = encodeWAV(channelData, sampleRate, 1);
    
    finalResults.push({
      stemName: modelConfig.stems[stemIdx],
      audioBuffer: buf,
      wavData,
    });
  }

  onProgress?.({ stage: 'done', progress: 100, message: 'Done!' });
  
  return finalResults;
}

function extractMask(
  data: Float32Array,
  numFreqs: number,
  numFrames: number
): Float32Array {
  const totalElements = numFreqs * numFrames;
  const mask = new Float32Array(totalElements);
  
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < Math.min(data.length, totalElements); i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  
  const range = max - min || 1;
  for (let i = 0; i < totalElements; i++) {
    const val = i < data.length ? data[i] : 0;
    mask[i] = (val - min) / range;
  }
  
  return mask;
}

function extractMultiMask(
  data: Float32Array,
  numStems: number,
  numFreqs: number,
  numFrames: number
): Float32Array[] {
  const frameSize = numFreqs * numFrames;
  const masks: Float32Array[] = [];
  
  const stemDataSize = Math.floor(data.length / numStems);
  
  for (let s = 0; s < numStems; s++) {
    const stemMask = new Float32Array(frameSize);
    const offset = s * stemDataSize;
    
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < frameSize && (offset + i) < data.length; i++) {
      const val = data[offset + i];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    
    const range = max - min || 1;
    for (let i = 0; i < frameSize; i++) {
      const val = (offset + i) < data.length ? data[offset + i] : 0;
      stemMask[i] = Math.max(0, Math.min(1, (val - min) / range));
    }
    
    masks.push(stemMask);
  }
  
  return masks;
}

function reshapeMask(flat: Float32Array, numFrames: number, numFreqs: number): Float32Array[] {
  const result: Float32Array[] = [];
  for (let t = 0; t < numFrames; t++) {
    const row = new Float32Array(numFreqs);
    for (let f = 0; f < numFreqs; f++) {
      const idx = t * numFreqs + f;
      row[f] = idx < flat.length ? flat[idx] : 0;
    }
    result.push(row);
  }
  return result;
}

function createFallbackMasks(
  numFrames: number,
  numFreqs: number,
  numStems: number
): Float32Array[] {
  const masks: Float32Array[] = [];
  const totalSize = numFrames * numFreqs;
  
  for (let s = 0; s < numStems; s++) {
    const flat = new Float32Array(totalSize);
    
    for (let t = 0; t < numFrames; t++) {
      for (let f = 0; f < numFreqs; f++) {
        const freqRatio = f / numFreqs;
        const idx = t * numFreqs + f;
        
        if (numStems === 2) {
          if (s === 0) {
            // Vocal mask: peak around 0.02-0.1 of spectrum (300Hz-4kHz)
            const center = 0.06;
            const width = 0.08;
            flat[idx] = Math.exp(-Math.pow((freqRatio - center) / width, 2));
          } else {
            const center = 0.06;
            const width = 0.08;
            flat[idx] = 1 - Math.exp(-Math.pow((freqRatio - center) / width, 2));
          }
        } else {
          const bandStart = s / numStems;
          const bandEnd = (s + 1) / numStems;
          const bandCenter = (bandStart + bandEnd) / 2;
          const bandWidth = (bandEnd - bandStart) * 0.8;
          flat[idx] = Math.exp(-Math.pow((freqRatio - bandCenter) / bandWidth, 2));
        }
      }
    }
    
    masks.push(flat);
  }
  
  return masks;
}
