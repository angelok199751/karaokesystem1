/**
 * Post-processing: convert Basic Pitch model outputs to MIDI notes
 */
import { BasicPitchOutput, BASIC_PITCH_CONFIG } from './basicPitchModel';

export interface MidiNote {
  startFrame: number;
  endFrame: number;
  pitch: number;  // MIDI note number (21-108)
  amplitude: number;  // Velocity (0-127)
}

export interface TranscriptionResult {
  notes: MidiNote[];
  sampleRate: number;
  hopSize: number;  // Samples per frame
}

/**
 * Convert model outputs to MIDI notes
 */
export function postprocessModelOutputs(
  outputs: BasicPitchOutput[],
  numWindows: number
): TranscriptionResult {
  const { NUM_FREQ_BINS, NUM_FRAMES, SAMPLE_RATE } = BASIC_PITCH_CONFIG;
  
  // Combine outputs from all windows
  const totalFrames = NUM_FRAMES * numWindows;
  const noteActivations = new Float32Array(totalFrames * NUM_FREQ_BINS);
  const onsetActivations = new Float32Array(totalFrames * NUM_FREQ_BINS);
  
  for (let w = 0; w < numWindows; w++) {
    const output = outputs[w];
    const frameOffset = w * NUM_FRAMES;
    
    for (let t = 0; t < NUM_FRAMES; t++) {
      for (let f = 0; f < NUM_FREQ_BINS; f++) {
        const idx = t * NUM_FREQ_BINS + f;
        const globalIdx = (frameOffset + t) * NUM_FREQ_BINS + f;
        
        noteActivations[globalIdx] = output.note[idx];
        onsetActivations[globalIdx] = output.onset[idx];
      }
    }
  }
  
  // Extract notes using onset detection
  const notes = extractNotes(noteActivations, onsetActivations, totalFrames, NUM_FREQ_BINS);
  
  // Calculate hop size (samples per frame)
  const hopSize = Math.floor(SAMPLE_RATE / 250);  // ~88 samples per frame at 22050 Hz
  
  return {
    notes,
    sampleRate: SAMPLE_RATE,
    hopSize
  };
}

/**
 * Extract MIDI notes from activation matrices
 */
function extractNotes(
  noteActivations: Float32Array,
  onsetActivations: Float32Array,
  numFrames: number,
  numPitches: number
): MidiNote[] {
  const NOTE_THRESHOLD = 0.5;
  const ONSET_THRESHOLD = 0.5;
  const MIN_NOTE_LENGTH = 2;  // Minimum frames (~8ms)
  
  const notes: MidiNote[] = [];
  
  // For each pitch
  for (let pitch = 0; pitch < numPitches; pitch++) {
    let inNote = false;
    let noteStart = 0;
    let maxAmplitude = 0;
    
    // Scan through frames
    for (let frame = 0; frame < numFrames; frame++) {
      const idx = frame * numPitches + pitch;
      const noteActive = noteActivations[idx] > NOTE_THRESHOLD;
      const onsetActive = onsetActivations[idx] > ONSET_THRESHOLD;
      
      if (onsetActive && !inNote) {
        // Start new note
        inNote = true;
        noteStart = frame;
        maxAmplitude = noteActivations[idx];
      } else if (noteActive && inNote) {
        // Continue note
        maxAmplitude = Math.max(maxAmplitude, noteActivations[idx]);
      } else if (!noteActive && inNote) {
        // End note
        inNote = false;
        const noteLength = frame - noteStart;
        
        if (noteLength >= MIN_NOTE_LENGTH) {
          notes.push({
            startFrame: noteStart,
            endFrame: frame,
            pitch: pitch + 21,  // Convert to MIDI note (21-108)
            amplitude: Math.min(127, Math.floor(maxAmplitude * 127))
          });
        }
      }
    }
    
    // Handle note that extends to the end
    if (inNote) {
      const noteLength = numFrames - noteStart;
      if (noteLength >= MIN_NOTE_LENGTH) {
        notes.push({
          startFrame: noteStart,
          endFrame: numFrames,
          pitch: pitch + 21,
          amplitude: Math.min(127, Math.floor(maxAmplitude * 127))
        });
      }
    }
  }
  
  return notes;
}

/**
 * Convert frames to time in seconds
 */
export function frameToTime(frame: number, hopSize: number, sampleRate: number): number {
  return (frame * hopSize) / sampleRate;
}
