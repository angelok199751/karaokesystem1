/**
 * Enhanced audio separation using classical signal processing
 * Combines Center Channel Extraction + Frequency-based filtering + Spectral analysis
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

// ── FFT Implementation ─────────────────────────────────────────────────────

function fft(re: Float32Array, im: Float32Array, inverse: boolean = false): void {
  const N = re.length;
  
  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < N - 1; i++) {
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
    let k = N >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Cooley-Tukey FFT
  for (let size = 2; size <= N; size *= 2) {
    const halfSize = size / 2;
    const angle = (inverse ? 1 : -1) * (2 * Math.PI) / size;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < N; i += size) {
      let curRe = 1;
      let curIm = 0;

      for (let k = 0; k < halfSize; k++) {
        const idx1 = i + k;
        const idx2 = i + k + halfSize;

        const tRe = curRe * re[idx2] - curIm * im[idx2];
        const tIm = curRe * im[idx2] + curIm * re[idx2];

        re[idx2] = re[idx1] - tRe;
        im[idx2] = im[idx1] - tIm;
        re[idx1] = re[idx1] + tRe;
        im[idx1] = im[idx1] + tIm;

        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < N; i++) {
      re[i] /= N;
      im[i] /= N;
    }
  }
}

// ── STFT/iSTFT ─────────────────────────────────────────────────────────────

function hannWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return window;
}

interface STFTResult {
  real: Float32Array[];
  imag: Float32Array[];
  numFrames: number;
  numFreqs: number;
}

function stft(signal: Float32Array, nFft: number, hopLength: number): STFTResult {
  const window = hannWindow(nFft);
  const numFrames = Math.floor((signal.length - nFft) / hopLength) + 1;
  const numFreqs = Math.floor(nFft / 2) + 1;
  
  const real: Float32Array[] = [];
  const imag: Float32Array[] = [];

  for (let frame = 0; frame < numFrames; frame++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    const start = frame * hopLength;

    for (let i = 0; i < nFft && (start + i) < signal.length; i++) {
      frameRe[i] = signal[start + i] * window[i];
    }

    fft(frameRe, frameIm, false);

    const posRe = new Float32Array(numFreqs);
    const posIm = new Float32Array(numFreqs);
    for (let i = 0; i < numFreqs; i++) {
      posRe[i] = frameRe[i];
      posIm[i] = frameIm[i];
    }

    real.push(posRe);
    imag.push(posIm);
  }

  return { real, imag, numFrames, numFreqs };
}

function istft(stftResult: STFTResult, nFft: number, hopLength: number, outputLength: number): Float32Array {
  const { real, imag, numFrames } = stftResult;
  const window = hannWindow(nFft);
  
  const output = new Float32Array(outputLength);
  const windowSum = new Float32Array(outputLength);

  for (let frame = 0; frame < numFrames; frame++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    const start = frame * hopLength;

    const numFreqs = real[frame].length;
    for (let i = 0; i < numFreqs; i++) {
      frameRe[i] = real[frame][i];
      frameIm[i] = imag[frame][i];
    }
    for (let i = 1; i < nFft - numFreqs + 1; i++) {
      frameRe[nFft - i] = real[frame][i];
      frameIm[nFft - i] = -imag[frame][i];
    }

    fft(frameRe, frameIm, true);

    for (let i = 0; i < nFft && (start + i) < outputLength; i++) {
      output[start + i] += frameRe[i] * window[i];
      windowSum[start + i] += window[i] * window[i];
    }
  }

  for (let i = 0; i < outputLength; i++) {
    if (windowSum[i] > 1e-8) {
      output[i] /= windowSum[i];
    }
  }

  return output;
}

// ── Enhanced Separation ────────────────────────────────────────────────────

/**
 * Enhanced separation using:
 * 1. Center Channel Extraction (base)
 * 2. Frequency-based spectral masking
 * 3. Phase correlation analysis
 */
export async function separateAudio(
  audioBuffer: AudioBuffer,
  onProgress?: (progress: SeparationProgress) => void
): Promise<SeparationResult[]> {
  onProgress?.({ stage: 'preparing', progress: 0, message: 'Analyzing audio...' });

  if (audioBuffer.numberOfChannels < 2) {
    throw new Error('Enhanced separation requires stereo audio. Please use a stereo audio file.');
  }

  const sampleRate = audioBuffer.sampleRate;
  const length = audioBuffer.length;
  
  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.getChannelData(1);

  onProgress?.({ stage: 'processing', progress: 10, message: 'Computing spectrograms...' });

  // STFT parameters
  const nFft = 2048;
  const hopLength = 512;

  // Compute STFT for both channels
  const stftL = stft(left, nFft, hopLength);
  const stftR = stft(right, nFft, hopLength);

  onProgress?.({ stage: 'processing', progress: 30, message: 'Analyzing phase correlation...' });

  // Compute magnitude and phase for both channels
  const numFrames = stftL.numFrames;
  const numFreqs = stftL.numFreqs;

  const magL: Float32Array[] = [];
  const magR: Float32Array[] = [];
  const phaseL: Float32Array[] = [];
  const phaseR: Float32Array[] = [];

  for (let frame = 0; frame < numFrames; frame++) {
    const mRe = new Float32Array(numFreqs);
    const mIm = new Float32Array(numFreqs);
    const pRe = new Float32Array(numFreqs);
    const pIm = new Float32Array(numFreqs);

    for (let freq = 0; freq < numFreqs; freq++) {
      const reL = stftL.real[frame][freq];
      const imL = stftL.imag[frame][freq];
      const reR = stftR.real[frame][freq];
      const imR = stftR.imag[frame][freq];

      mRe[freq] = Math.sqrt(reL * reL + imL * imL);
      mIm[freq] = Math.sqrt(reR * reR + imR * imR);
      pRe[freq] = Math.atan2(imL, reL);
      pIm[freq] = Math.atan2(imR, reR);
    }

    magL.push(mRe);
    magR.push(mIm);
    phaseL.push(pRe);
    phaseR.push(pIm);
  }

  onProgress?.({ stage: 'processing', progress: 50, message: 'Creating vocal mask...' });

  // Create vocal mask using multiple criteria:
  // 1. Center channel (L+R similarity)
  // 2. Phase correlation (similar phase = center)
  // 3. Frequency weighting (vocals typically 200Hz-4kHz)
  
  const vocalMask: Float32Array[] = [];
  const nyquist = sampleRate / 2;

  for (let frame = 0; frame < numFrames; frame++) {
    const mask = new Float32Array(numFreqs);

    for (let freq = 0; freq < numFreqs; freq++) {
      const freqHz = (freq / numFreqs) * nyquist;
      
      // 1. Center channel strength (magnitude similarity)
      const magAvg = (magL[frame][freq] + magR[frame][freq]) / 2;
      const magDiff = Math.abs(magL[frame][freq] - magR[frame][freq]);
      const centerStrength = magAvg > 0 ? 1 - (magDiff / (magAvg + 1e-10)) : 0;
      
      // 2. Phase correlation (similar phase = center)
      const phaseDiff = Math.abs(phaseL[frame][freq] - phaseR[frame][freq]);
      const phaseCorr = Math.cos(phaseDiff); // 1 = same phase, -1 = opposite
      
      // 3. Frequency weighting for vocals (200Hz - 4kHz)
      let freqWeight = 0;
      if (freqHz >= 200 && freqHz <= 4000) {
        // Smooth bell curve centered around 1kHz
        const center = 1000;
        const width = 1500;
        freqWeight = Math.exp(-Math.pow((freqHz - center) / width, 2));
      } else if (freqHz < 200) {
        // Low frequencies - less likely to be vocals
        freqWeight = 0.2;
      } else {
        // High frequencies - less likely to be vocals
        freqWeight = 0.3;
      }

      // Combine all criteria
      const combinedMask = (centerStrength * 0.5 + phaseCorr * 0.3 + 0.2) * (0.5 + freqWeight * 0.5);
      mask[freq] = Math.max(0, Math.min(1, combinedMask));
    }

    vocalMask.push(mask);
  }

  onProgress?.({ stage: 'processing', progress: 70, message: 'Applying masks and reconstructing...' });

  // Apply masks to create separated spectrograms
  const vocalsReal: Float32Array[] = [];
  const vocalsImag: Float32Array[] = [];
  const instrReal: Float32Array[] = [];
  const instrImag: Float32Array[] = [];

  for (let frame = 0; frame < numFrames; frame++) {
    const vRe = new Float32Array(numFreqs);
    const vIm = new Float32Array(numFreqs);
    const iRe = new Float32Array(numFreqs);
    const iIm = new Float32Array(numFreqs);

    for (let freq = 0; freq < numFreqs; freq++) {
      const mask = vocalMask[frame][freq];
      
      // Center channel for vocals
      const centerRe = (stftL.real[frame][freq] + stftR.real[frame][freq]) / 2;
      const centerIm = (stftL.imag[frame][freq] + stftR.imag[frame][freq]) / 2;
      
      // Apply vocal mask
      vRe[freq] = centerRe * mask;
      vIm[freq] = centerIm * mask;
      
      // Instrumental = original - vocals
      const avgRe = (stftL.real[frame][freq] + stftR.real[frame][freq]) / 2;
      const avgIm = (stftL.imag[frame][freq] + stftR.imag[frame][freq]) / 2;
      iRe[freq] = avgRe - vRe[freq];
      iIm[freq] = avgIm - vIm[freq];
    }

    vocalsReal.push(vRe);
    vocalsImag.push(vIm);
    instrReal.push(iRe);
    instrImag.push(iIm);
  }

  // Reconstruct audio using iSTFT
  const vocalsMono = istft({ real: vocalsReal, imag: vocalsImag, numFrames, numFreqs }, nFft, hopLength, length);
  const instrMono = istft({ real: instrReal, imag: instrImag, numFrames, numFreqs }, nFft, hopLength, length);

  onProgress?.({ stage: 'reconstructing', progress: 90, message: 'Building stereo output...' });

  // Create stereo output (duplicate mono to both channels)
  const vocalsL = new Float32Array(vocalsMono.length);
  vocalsL.set(vocalsMono);
  const vocalsR = new Float32Array(vocalsMono.length);
  vocalsR.set(vocalsMono);
  const instrL = new Float32Array(instrMono.length);
  instrL.set(instrMono);
  const instrR = new Float32Array(instrMono.length);
  instrR.set(instrMono);

  // Create AudioBuffers
  const vocalsCtx = new OfflineAudioContext(2, length, sampleRate);
  const vocalsBuf = vocalsCtx.createBuffer(2, length, sampleRate);
  vocalsBuf.copyToChannel(vocalsL, 0);
  vocalsBuf.copyToChannel(vocalsR, 1);

  const instrCtx = new OfflineAudioContext(2, length, sampleRate);
  const instrBuf = instrCtx.createBuffer(2, length, sampleRate);
  instrBuf.copyToChannel(instrL, 0);
  instrBuf.copyToChannel(instrR, 1);

  onProgress?.({ stage: 'reconstructing', progress: 95, message: 'Encoding WAV files...' });

  // Encode to WAV
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
