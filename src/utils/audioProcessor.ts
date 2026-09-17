/**
 * Audio processing utilities: STFT, iSTFT, WAV encoding/decoding
 */

// Generate Hann window
export function hannWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return window;
}

// Simple DFT (for small frames) - used as fallback
export function dft(re: Float32Array, im: Float32Array, inverse: boolean = false): void {
  const N = re.length;
  const newRe = new Float32Array(N);
  const newIm = new Float32Array(N);
  const sign = inverse ? 1 : -1;
  const factor = inverse ? 1 / N : 1;

  for (let k = 0; k < N; k++) {
    let sumRe = 0;
    let sumIm = 0;
    for (let n = 0; n < N; n++) {
      const angle = (sign * 2 * Math.PI * k * n) / N;
      sumRe += re[n] * Math.cos(angle) - im[n] * Math.sin(angle);
      sumIm += re[n] * Math.sin(angle) + im[n] * Math.cos(angle);
    }
    newRe[k] = sumRe * factor;
    newIm[k] = sumIm * factor;
  }

  for (let i = 0; i < N; i++) {
    re[i] = newRe[i];
    im[i] = newIm[i];
  }
}

// FFT using Cooley-Tukey radix-2 algorithm
export function fft(re: Float32Array, im: Float32Array, inverse: boolean = false): void {
  const N = re.length;
  
  // Check if N is power of 2
  if ((N & (N - 1)) !== 0) {
    dft(re, im, inverse);
    return;
  }

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

  // Cooley-Tukey iterative FFT
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

// STFT parameters
export interface STFTParams {
  nFft: number;
  hopLength: number;
  winLength: number;
}

// Compute STFT of a mono signal
export function computeSTFT(
  signal: Float32Array,
  params: STFTParams
): { real: Float32Array[]; imag: Float32Array[] } {
  const { nFft, hopLength, winLength } = params;
  const window = hannWindow(winLength);
  
  const numFrames = Math.floor((signal.length - nFft) / hopLength) + 1;
  const numFreqs = Math.floor(nFft / 2) + 1;
  
  const real: Float32Array[] = [];
  const imag: Float32Array[] = [];

  for (let frame = 0; frame < numFrames; frame++) {
    const frameRe = new Float32Array(nFft);
    const frameIm = new Float32Array(nFft);
    const start = frame * hopLength;

    for (let i = 0; i < winLength && (start + i) < signal.length; i++) {
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

  return { real, imag };
}

// Compute iSTFT from spectrogram
export function computeISTFT(
  real: Float32Array[],
  imag: Float32Array[],
  params: STFTParams,
  outputLength: number
): Float32Array {
  const { nFft, hopLength, winLength } = params;
  const window = hannWindow(winLength);
  const numFrames = real.length;
  
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

    for (let i = 0; i < winLength && (start + i) < outputLength; i++) {
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

// Encode Float32Array to WAV ArrayBuffer
export function encodeWAV(audioData: Float32Array, sampleRate: number, numChannels: number = 1): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + audioData.length * 2);
  const view = new DataView(buffer);

  // WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + audioData.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, audioData.length * 2, true);

  // Write samples
  let offset = 44;
  for (let i = 0; i < audioData.length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    offset += 2;
  }

  return buffer;
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

// Decode AudioBuffer to mono Float32Array
export function audioBufferToMono(audioBuffer: AudioBuffer): Float32Array {
  const numChannels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const mono = new Float32Array(length);

  if (numChannels === 1) {
    mono.set(audioBuffer.getChannelData(0));
  } else {
    for (let ch = 0; ch < numChannels; ch++) {
      const channelData = audioBuffer.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        mono[i] += channelData[i] / numChannels;
      }
    }
  }

  return mono;
}

// Compute magnitude spectrogram
export function computeMagnitude(real: Float32Array[], imag: Float32Array[]): Float32Array[] {
  return real.map((frameRe, i) => {
    const frameIm = imag[i];
    const mag = new Float32Array(frameRe.length);
    for (let j = 0; j < frameRe.length; j++) {
      mag[j] = Math.sqrt(frameRe[j] * frameRe[j] + frameIm[j] * frameIm[j]);
    }
    return mag;
  });
}
