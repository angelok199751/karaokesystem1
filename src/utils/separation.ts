/**
 * Audio separation using classical signal processing methods
 * No ML model required - works instantly!
 */

export interface SeparationResult {
  stemName: string;
  audioBuffer: AudioBuffer;
  wavData: ArrayBuffer;
}

export interface SeparationProgress {
  stage: 'preparing' | 'processing' | 'reconstructing' | 'done';
  progress: number;
  message: string;
}

/**
 * Center Channel Extraction method
 * Extracts vocals (center) and instrumental (sides) from stereo audio
 */
export async function separateAudio(
  audioBuffer: AudioBuffer,
  onProgress?: (progress: SeparationProgress) => void
): Promise<SeparationResult[]> {
  onProgress?.({ stage: 'preparing', progress: 0, message: 'Preparing audio...' });

  // Check if audio is stereo
  if (audioBuffer.numberOfChannels < 2) {
    throw new Error('Center Channel Extraction requires stereo audio. Please use a stereo audio file.');
  }

  const sampleRate = audioBuffer.sampleRate;
  const length = audioBuffer.length;
  
  // Get left and right channels
  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.getChannelData(1);

  onProgress?.({ stage: 'processing', progress: 20, message: 'Extracting center channel (vocals)...' });

  // Extract center channel (vocals)
  // Center = (L + R) / 2
  const vocalsL = new Float32Array(length);
  const vocalsR = new Float32Array(length);
  
  for (let i = 0; i < length; i++) {
    const center = (left[i] + right[i]) / 2;
    vocalsL[i] = center;
    vocalsR[i] = center;
  }

  onProgress?.({ stage: 'processing', progress: 50, message: 'Extracting side channels (instrumental)...' });

  // Extract side channels (instrumental)
  // Side = (L - R) / 2 for left, (R - L) / 2 for right
  // Then reconstruct: instrumental = original - vocals
  const instrL = new Float32Array(length);
  const instrR = new Float32Array(length);
  
  for (let i = 0; i < length; i++) {
    // Remove center from each channel
    const center = (left[i] + right[i]) / 2;
    instrL[i] = left[i] - center;
    instrR[i] = right[i] - center;
  }

  onProgress?.({ stage: 'reconstructing', progress: 80, message: 'Building audio buffers...' });

  // Create AudioBuffers
  const vocalsCtx = new OfflineAudioContext(2, length, sampleRate);
  const vocalsBuf = vocalsCtx.createBuffer(2, length, sampleRate);
  vocalsBuf.copyToChannel(vocalsL, 0);
  vocalsBuf.copyToChannel(vocalsR, 1);

  const instrCtx = new OfflineAudioContext(2, length, sampleRate);
  const instrBuf = instrCtx.createBuffer(2, length, sampleRate);
  instrBuf.copyToChannel(instrL, 0);
  instrBuf.copyToChannel(instrR, 1);

  onProgress?.({ stage: 'reconstructing', progress: 90, message: 'Encoding WAV files...' });

  // Encode to WAV (stereo)
  const vocalsStereo = interleaveStereo(vocalsL, vocalsR);
  const instrStereo = interleaveStereo(instrL, instrR);
  
  const vocalsWav = encodeStereoWAV(vocalsStereo, sampleRate);
  const instrWav = encodeStereoWAV(instrStereo, sampleRate);

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
