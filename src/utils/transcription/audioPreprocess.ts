/**
 * Audio preprocessing for Basic Pitch
 * Resample to 22050 Hz, mono, and split into overlapping windows
 */
import { BASIC_PITCH_CONFIG } from './basicPitchModel';

/**
 * Resample audio to target sample rate using OfflineAudioContext
 */
export async function resampleAudio(
  audioBuffer: AudioBuffer,
  targetSampleRate: number
): Promise<AudioBuffer> {
  if (audioBuffer.sampleRate === targetSampleRate) {
    return audioBuffer;
  }

  const offlineCtx = new OfflineAudioContext(
    1,
    Math.ceil(audioBuffer.duration * targetSampleRate),
    targetSampleRate
  );

  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start();

  return await offlineCtx.startRendering();
}

/**
 * Convert stereo audio to mono by averaging channels
 */
export function stereoToMono(audioBuffer: AudioBuffer): Float32Array {
  if (audioBuffer.numberOfChannels === 1) {
    return audioBuffer.getChannelData(0);
  }

  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.getChannelData(1);
  const mono = new Float32Array(left.length);

  for (let i = 0; i < left.length; i++) {
    mono[i] = (left[i] + right[i]) / 2;
  }

  return mono;
}

/**
 * Split audio into overlapping windows for Basic Pitch
 */
export function splitIntoWindows(
  audio: Float32Array,
  windowSize: number = BASIC_PITCH_CONFIG.WINDOW_SIZE,
  overlap: number = BASIC_PITCH_CONFIG.OVERLAP
): Float32Array[] {
  const windows: Float32Array[] = [];
  const step = windowSize - overlap;

  for (let start = 0; start < audio.length; start += step) {
    const end = Math.min(start + windowSize, audio.length);
    const window = new Float32Array(windowSize);
    
    // Copy audio data
    const copyLength = end - start;
    window.set(audio.subarray(start, end), 0);
    
    // Pad with zeros if needed
    if (copyLength < windowSize) {
      // Already initialized with zeros
    }
    
    windows.push(window);
    
    // Stop if we've reached the end
    if (end >= audio.length) {
      break;
    }
  }

  return windows;
}

/**
 * Prepare audio for Basic Pitch inference
 * Resample to 22050 Hz, convert to mono, split into windows
 */
export async function prepareAudioForTranscription(
  audioBuffer: AudioBuffer
): Promise<Float32Array[]> {
  // Resample to 22050 Hz
  const resampled = await resampleAudio(audioBuffer, BASIC_PITCH_CONFIG.SAMPLE_RATE);
  
  // Convert to mono
  const mono = stereoToMono(resampled);
  
  // Split into overlapping windows
  const windows = splitIntoWindows(mono);
  
  return windows;
}
