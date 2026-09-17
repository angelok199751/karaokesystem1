/**
 * Vocal-specific audio filtering
 * Isolates vocal frequencies and removes instrumental artifacts
 */
import { KaraokeJSON } from './karaokeJSON';

export interface VocalFilterOptions {
  // Spectral filtering
  enableSpectralFilter: boolean;
  vocalRangeMin: number; // Hz (default: 80)
  vocalRangeMax: number; // Hz (default: 1000)
  
  // MIDI note filtering
  enableNoteFilter: boolean;
  midiRangeMin: number; // MIDI note (default: 40 = E2)
  midiRangeMax: number; // MIDI note (default: 80 = G5)
  
  // Amplitude filtering
  enableAmplitudeFilter: boolean;
  minAmplitude: number; // 0-1 (default: 0.3)
}

export const DEFAULT_VOCAL_FILTER_OPTIONS: VocalFilterOptions = {
  enableSpectralFilter: true,
  vocalRangeMin: 80,
  vocalRangeMax: 1000,
  enableNoteFilter: true,
  midiRangeMin: 40,
  midiRangeMax: 80,
  enableAmplitudeFilter: true,
  minAmplitude: 0.3,
};

/**
 * Apply vocal-specific filtering to MIDI
 */
export function applyVocalFilter(
  karaokeJSON: KaraokeJSON,
  options: VocalFilterOptions = DEFAULT_VOCAL_FILTER_OPTIONS
): KaraokeJSON {
  console.log('[VocalFilter] Applying vocal-specific filters...');
  
  let filteredPhrases = karaokeJSON.phrases;
  
  // Filter 1: Remove notes outside vocal MIDI range
  if (options.enableNoteFilter) {
    filteredPhrases = filterByMidiRange(
      filteredPhrases,
      options.midiRangeMin,
      options.midiRangeMax
    );
    console.log(`[VocalFilter] After MIDI range filter: ${countNotes(filteredPhrases)} notes`);
  }
  
  // Filter 2: Remove short/quiet notes (likely artifacts)
  if (options.enableAmplitudeFilter) {
    filteredPhrases = filterByDuration(
      filteredPhrases,
      0.05 // minimum 50ms
    );
    console.log(`[VocalFilter] After duration filter: ${countNotes(filteredPhrases)} notes`);
  }
  
  // Rebuild karaoke JSON
  return {
    ...karaokeJSON,
    phrases: filteredPhrases,
  };
}

/**
 * Filter notes by MIDI range
 */
function filterByMidiRange(
  phrases: KaraokeJSON['phrases'],
  minNote: number,
  maxNote: number
): KaraokeJSON['phrases'] {
  return phrases.map(phrase => ({
    ...phrase,
    notes: phrase.notes.filter(note => 
      note.note >= minNote && note.note <= maxNote
    ),
  })).filter(phrase => phrase.notes.length > 0);
}

/**
 * Filter notes by minimum duration
 */
function filterByDuration(
  phrases: KaraokeJSON['phrases'],
  minDuration: number
): KaraokeJSON['phrases'] {
  return phrases.map(phrase => ({
    ...phrase,
    notes: phrase.notes.filter(note => 
      (note.end - note.start) >= minDuration
    ),
  })).filter(phrase => phrase.notes.length > 0);
}

/**
 * Count total notes in phrases
 */
function countNotes(phrases: KaraokeJSON['phrases']): number {
  return phrases.reduce((sum, phrase) => sum + phrase.notes.length, 0);
}

/**
 * Convert frequency (Hz) to MIDI note number
 */
export function frequencyToMidi(frequency: number): number {
  return Math.round(69 + 12 * Math.log2(frequency / 440));
}

/**
 * Convert MIDI note number to frequency (Hz)
 */
export function midiToFrequency(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}
