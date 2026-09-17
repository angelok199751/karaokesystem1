/**
 * Audio preprocessing for Basic Pitch
 */
import { BASIC_PITCH_CONFIG } from './basicPitchModel';

/**
 * Resample audio to target sample rate
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
 * Based on Python implementation:
 * - Add padding at the beginning (overlap_len / 2)
 * - Split into windows of WINDOW_SIZE
 * - Hop size = WINDOW_SIZE - OVERLAP
 */
export function splitIntoWindows(
  audio: Float32Array,
  windowSize: number = BASIC_PITCH_CONFIG.WINDOW_SIZE,
  overlap: number = BASIC_PITCH_CONFIG.OVERLAP
): Float32Array[] {
  const windows: Float32Array[] = [];
  const hopSize = windowSize - overlap;
  const paddingSize = Math.floor(overlap / 2);
  
  // Add padding at the beginning (like Python code)
  const paddedAudio = new Float32Array(paddingSize + audio.length);
  paddedAudio.set(audio, paddingSize);
  
  console.log(`[AudioPreprocess] Audio length: ${audio.length}, Padded: ${paddedAudio.length}`);
  console.log(`[AudioPreprocess] Window size: ${windowSize}, Hop: ${hopSize}, Padding: ${paddingSize}`);

  // Split into windows
  for (let start = 0; start < paddedAudio.length; start += hopSize) {
    const end = Math.min(start + windowSize, paddedAudio.length);
    const window = new Float32Array(windowSize);
    
    // Copy audio data
    const copyLength = end - start;
    window.set(paddedAudio.subarray(start, end), 0);
    
    // Pad with zeros if needed (for the last window)
    if (copyLength < windowSize) {
      // Already initialized with zeros
    }
    
    windows.push(window);
    
    // Stop if we've reached the end
    if (end >= paddedAudio.length) {
      break;
    }
  }

  console.log(`[AudioPreprocess] Created ${windows.length} windows`);

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
