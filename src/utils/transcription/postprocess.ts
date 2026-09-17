/**
 * Post-processing: convert Basic Pitch model outputs to MIDI notes
 */
import { BasicPitchOutput, BASIC_PITCH_CONFIG } from './basicPitchModel';

export interface MidiNote {
  startFrame: number;
  endFrame: number;
  pitch: number;
  amplitude: number;
}

export interface TranscriptionResult {
  notes: MidiNote[];
  sampleRate: number;
  hopSize: number;
}

export function postprocessModelOutputs(
  outputs: BasicPitchOutput[],
  numWindows: number
): TranscriptionResult {
  const { NUM_FREQ_BINS, SAMPLE_RATE, FFT_HOP, OVERLAP } = BASIC_PITCH_CONFIG;
  
  if (outputs.length === 0) {
    return { notes: [], sampleRate: SAMPLE_RATE, hopSize: FFT_HOP };
  }
  
  const noteData = outputs[0].note;
  const totalElements = noteData.length;
  const framesPerWindow = Math.floor(totalElements / NUM_FREQ_BINS);
  
  console.log(`[PostProcess] Frames per window: ${framesPerWindow}, Freq bins: ${NUM_FREQ_BINS}`);
  
  const nOverlappingFrames = 30;
  const nOlap = Math.floor(nOverlappingFrames / 2);
  const usableFramesPerWindow = framesPerWindow - nOverlappingFrames;
  const totalFrames = usableFramesPerWindow * numWindows;
  
  const noteActivations = new Float32Array(totalFrames * NUM_FREQ_BINS);
  const onsetActivations = new Float32Array(totalFrames * NUM_FREQ_BINS);
  
  for (let w = 0; w < numWindows; w++) {
    const output = outputs[w];
    const frameOffset = w * usableFramesPerWindow;
    
    for (let t = nOlap; t < framesPerWindow - nOlap; t++) {
      const localT = t - nOlap;
      const globalT = frameOffset + localT;
      
      if (globalT >= totalFrames) break;
      
      for (let f = 0; f < NUM_FREQ_BINS; f++) {
        const srcIdx = t * NUM_FREQ_BINS + f;
        const dstIdx = globalT * NUM_FREQ_BINS + f;
        
        if (srcIdx < output.note.length) {
          noteActivations[dstIdx] = output.note[srcIdx];
        }
        if (srcIdx < output.onset.length) {
          onsetActivations[dstIdx] = output.onset[srcIdx];
        }
      }
    }
  }
  
  const notes = extractNotes(noteActivations, onsetActivations, totalFrames, NUM_FREQ_BINS);
  const hopSize = FFT_HOP;
  
  return { notes, sampleRate: SAMPLE_RATE, hopSize };
}

function extractNotes(
  noteActivations: Float32Array,
  onsetActivations: Float32Array,
  numFrames: number,
  numPitches: number
): MidiNote[] {
  const NOTE_THRESHOLD = 0.5;
  const ONSET_THRESHOLD = 0.5;
  const MIN_NOTE_LENGTH_FRAMES = 2;
  
  const notes: MidiNote[] = [];
  
  for (let pitch = 0; pitch < numPitches; pitch++) {
    let inNote = false;
    let noteStart = 0;
    let maxAmplitude = 0;
    
    for (let frame = 0; frame < numFrames; frame++) {
      const idx = frame * numPitches + pitch;
      const noteActive = noteActivations[idx] > NOTE_THRESHOLD;
      const onsetActive = onsetActivations[idx] > ONSET_THRESHOLD;
      
      if (onsetActive && !inNote) {
        inNote = true;
        noteStart = frame;
        maxAmplitude = noteActivations[idx];
      } else if (noteActive && inNote) {
        maxAmplitude = Math.max(maxAmplitude, noteActivations[idx]);
      } else if (!noteActive && inNote) {
        inNote = false;
        const noteLength = frame - noteStart;
        
        if (noteLength >= MIN_NOTE_LENGTH_FRAMES) {
          notes.push({
            startFrame: noteStart,
            endFrame: frame,
            pitch: pitch + 21,
            amplitude: Math.min(127, Math.floor(maxAmplitude * 127))
          });
        }
      }
    }
    
    if (inNote) {
      const noteLength = numFrames - noteStart;
      if (noteLength >= MIN_NOTE_LENGTH_FRAMES) {
        notes.push({
          startFrame: noteStart,
          endFrame: numFrames,
          pitch: pitch + 21,
          amplitude: Math.min(127, Math.floor(maxAmplitude * 127))
        });
      }
    }
  }
  
  console.log(`[PostProcess] Extracted ${notes.length} notes`);
  return notes;
}

export function frameToTime(frame: number, hopSize: number, sampleRate: number): number {
  return (frame * hopSize) / sampleRate;
}
