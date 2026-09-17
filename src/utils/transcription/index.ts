/**
 * Audio transcription module
 * Converts audio to MIDI using Basic Pitch model
 */
import { loadBasicPitchModel, runBasicPitchInference } from './basicPitchModel';
import { prepareAudioForTranscription } from './audioPreprocess';
import { postprocessModelOutputs, TranscriptionResult } from './postprocess';
import { generateMidiFile } from './midiEncoder';

export interface TranscriptionProgress {
  stage: 'loading' | 'preparing' | 'processing' | 'postprocessing' | 'encoding' | 'done';
  progress: number;
  message: string;
}

export interface TranscriptionOutput {
  midiData: Uint8Array;
  transcriptionResult: TranscriptionResult;
}

export async function transcribeAudio(
  audioBuffer: AudioBuffer,
  onProgress?: (progress: TranscriptionProgress) => void
): Promise<TranscriptionOutput> {
  // Load model
  onProgress?.({ stage: 'loading', progress: 0, message: 'Loading model...' });
  await loadBasicPitchModel();
  onProgress?.({ stage: 'loading', progress: 20, message: 'Model loaded' });

  // Prepare audio
  onProgress?.({ stage: 'preparing', progress: 20, message: 'Preparing audio...' });
  const windows = await prepareAudioForTranscription(audioBuffer);
  onProgress?.({ stage: 'preparing', progress: 40, message: `Created ${windows.length} windows` });

  // Process each window
  onProgress?.({ stage: 'processing', progress: 40, message: 'Processing audio...' });
  const outputs = [];
  
  for (let i = 0; i < windows.length; i++) {
    const progress = 40 + (i / windows.length) * 40;
    onProgress?.({ 
      stage: 'processing', 
      progress, 
      message: `Processing window ${i + 1}/${windows.length}` 
    });
    
    const output = await runBasicPitchInference(windows[i]);
    outputs.push(output);
  }
  
  onProgress?.({ stage: 'processing', progress: 80, message: 'Inference complete' });

  // Post-process
  onProgress?.({ stage: 'postprocessing', progress: 80, message: 'Extracting notes...' });
  const result = postprocessModelOutputs(outputs, windows.length);
  onProgress?.({ stage: 'postprocessing', progress: 90, message: `Extracted ${result.notes.length} notes` });

  // Generate MIDI
  onProgress?.({ stage: 'encoding', progress: 90, message: 'Generating MIDI...' });
  const midiData = generateMidiFile(result);
  onProgress?.({ stage: 'encoding', progress: 95, message: `MIDI file size: ${midiData.length} bytes` });

  onProgress?.({ stage: 'done', progress: 100, message: 'Transcription complete!' });

  return { midiData, transcriptionResult: result };
}

export function downloadMidi(midiData: Uint8Array, filename: string = 'transcription.mid'): void {
  const arrayBuffer = new ArrayBuffer(midiData.length);
  const view = new Uint8Array(arrayBuffer);
  view.set(midiData);
  
  const blob = new Blob([arrayBuffer], { type: 'audio/midi' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

export type { TranscriptionResult } from './postprocess';
export type { MidiNote } from './postprocess';

// Re-export karaoke JSON utilities
export { 
  generateKaraokeJSON, 
  exportKaraokeJSON, 
  downloadKaraokeJSON 
} from './karaokeJSON';
export type { KaraokeNote, KaraokePhrase, KaraokeJSON } from './karaokeJSON';

// Re-export MIDI cleaner
export { cleanMidi } from './midiCleaner';
export type { CleanerOptions } from './midiCleaner';

// Re-export lyrics aligner
export { 
  alignNotesToLyrics, 
  exportAlignedKaraoke, 
  downloadAlignedKaraoke 
} from './lyricsAligner';
export type { 
  LyricSegment, 
  LyricsFile, 
  AlignedNote, 
  AlignedSegment, 
  AlignedKaraokeFile 
} from './lyricsAligner';

// Re-export auto lyrics generator
export { 
  generateAutoLyrics, 
  exportAutoLyrics, 
  downloadAutoLyrics 
} from './autoLyricsGenerator';
export type { AutoLyricsOptions } from './autoLyricsGenerator';
