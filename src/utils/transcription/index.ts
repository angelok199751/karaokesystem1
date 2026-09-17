/**
 * Main transcription module
 * Orchestrates the full audio-to-MIDI pipeline
 */
import { loadBasicPitchModel, runBasicPitchInference, BasicPitchOutput } from './basicPitchModel';
import { prepareAudioForTranscription } from './audioPreprocess';
import { postprocessModelOutputs, TranscriptionResult } from './postprocess';
import { generateMidiFile } from './midiEncoder';

export interface TranscriptionProgress {
  stage: 'loading' | 'preparing' | 'inferring' | 'postprocessing' | 'encoding' | 'done';
  progress: number;  // 0-100
  message: string;
}

/**
 * Transcribe audio to MIDI
 */
export async function transcribeAudio(
  audioBuffer: AudioBuffer,
  onProgress?: (progress: TranscriptionProgress) => void
): Promise<Uint8Array> {
  // Stage 1: Load model
  onProgress?.({
    stage: 'loading',
    progress: 0,
    message: 'Loading Basic Pitch model...'
  });
  
  await loadBasicPitchModel();
  
  onProgress?.({
    stage: 'loading',
    progress: 10,
    message: 'Model loaded'
  });
  
  // Stage 2: Prepare audio
  onProgress?.({
    stage: 'preparing',
    progress: 15,
    message: 'Preparing audio...'
  });
  
  const windows = await prepareAudioForTranscription(audioBuffer);
  
  onProgress?.({
    stage: 'preparing',
    progress: 25,
    message: `Audio prepared: ${windows.length} windows`
  });
  
  // Stage 3: Run inference
  const outputs: BasicPitchOutput[] = [];
  
  for (let i = 0; i < windows.length; i++) {
    const progress = 25 + (i / windows.length) * 50;
    onProgress?.({
      stage: 'inferring',
      progress,
      message: `Transcribing window ${i + 1}/${windows.length}...`
    });
    
    const output = await runBasicPitchInference(windows[i]);
    outputs.push(output);
    
    // Yield to UI
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  
  onProgress?.({
    stage: 'inferring',
    progress: 75,
    message: 'Inference complete'
  });
  
  // Stage 4: Post-process
  onProgress?.({
    stage: 'postprocessing',
    progress: 80,
    message: 'Extracting notes...'
  });
  
  const transcription = postprocessModelOutputs(outputs, windows.length);
  
  onProgress?.({
    stage: 'postprocessing',
    progress: 85,
    message: `Found ${transcription.notes.length} notes`
  });
  
  // Stage 5: Generate MIDI
  onProgress?.({
    stage: 'encoding',
    progress: 90,
    message: 'Generating MIDI file...'
  });
  
  const midiData = generateMidiFile(transcription);
  
  onProgress?.({
    stage: 'done',
    progress: 100,
    message: 'MIDI file ready!'
  });
  
  return midiData;
}

/**
 * Download MIDI file
 */
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
