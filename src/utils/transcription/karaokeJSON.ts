/**
 * Karaoke JSON generator
 * Converts MIDI notes to karaoke-friendly JSON format with phrase grouping
 */
import { MidiNote, TranscriptionResult, frameToTime } from './postprocess';

export interface KaraokeNote {
  start: number;
  end: number;
  note: number;
}

export interface KaraokePhrase {
  notes: KaraokeNote[];
  start: number;
  end: number;
}

export interface KaraokeJSON {
  phrases: KaraokePhrase[];
  bpm: number;
  timeSignature: [number, number];
}

// Параметры для группировки по фразам
const PHRASE_GAP_THRESHOLD = 0.5; // секунды паузы между фразами
const TIME_ROUNDING_PRECISION = 3; // количество знаков после запятой

/**
 * Конвертирует MidiNote в KaraokeNote с временем в секундах
 */
function midiNoteToKaraokeNote(
  midiNote: MidiNote,
  hopSize: number,
  sampleRate: number
): KaraokeNote {
  const start = frameToTime(midiNote.startFrame, hopSize, sampleRate);
  const end = frameToTime(midiNote.endFrame, hopSize, sampleRate);
  
  return {
    start: roundTime(start),
    end: roundTime(end),
    note: midiNote.pitch
  };
}

/**
 * Округляет время до нужной точности
 */
function roundTime(time: number): number {
  const multiplier = Math.pow(10, TIME_ROUNDING_PRECISION);
  return Math.round(time * multiplier) / multiplier;
}

/**
 * Группирует ноты по фразам на основе пауз
 */
function groupNotesIntoPhrases(notes: KaraokeNote[]): KaraokePhrase[] {
  if (notes.length === 0) return [];
  
  // Сортируем ноты по времени начала
  const sortedNotes = [...notes].sort((a, b) => a.start - b.start);
  
  const phrases: KaraokePhrase[] = [];
  let currentPhrase: KaraokeNote[] = [sortedNotes[0]];
  
  for (let i = 1; i < sortedNotes.length; i++) {
    const prevNote = sortedNotes[i - 1];
    const currentNote = sortedNotes[i];
    
    // Проверяем паузу между нотами
    const gap = currentNote.start - prevNote.end;
    
    if (gap > PHRASE_GAP_THRESHOLD) {
      // Начинаем новую фразу
      phrases.push({
        notes: currentPhrase,
        start: currentPhrase[0].start,
        end: currentPhrase[currentPhrase.length - 1].end
      });
      currentPhrase = [currentNote];
    } else {
      // Добавляем в текущую фразу
      currentPhrase.push(currentNote);
    }
  }
  
  // Добавляем последнюю фразу
  if (currentPhrase.length > 0) {
    phrases.push({
      notes: currentPhrase,
      start: currentPhrase[0].start,
      end: currentPhrase[currentPhrase.length - 1].end
    });
  }
  
  return phrases;
}

/**
 * Оценивает BPM на основе средней длительности нот
 */
function estimateBPM(notes: KaraokeNote[]): number {
  if (notes.length === 0) return 120; // дефолтное значение
  
  // Считаем среднюю длительность нот
  const durations = notes.map(n => n.end - n.start);
  const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
  
  // Примерная оценка BPM (очень грубая)
  // Предполагаем, что средняя нота = четверть
  const bpm = Math.round(60 / avgDuration);
  
  // Ограничиваем разумными значениями
  return Math.max(60, Math.min(200, bpm));
}

/**
 * Генерирует KaraokeJSON из результата транскрипции
 */
export function generateKaraokeJSON(
  transcription: TranscriptionResult
): KaraokeJSON {
  const { notes, sampleRate, hopSize } = transcription;
  
  // Конвертируем в KaraokeNote
  const karaokeNotes = notes.map(note => 
    midiNoteToKaraokeNote(note, hopSize, sampleRate)
  );
  
  // Группируем по фразам
  const phrases = groupNotesIntoPhrases(karaokeNotes);
  
  // Оцениваем BPM
  const bpm = estimateBPM(karaokeNotes);
  
  return {
    phrases,
    bpm,
    timeSignature: [4, 4] // дефолтная метрика 4/4
  };
}

/**
 * Экспортирует KaraokeJSON в строку
 */
export function exportKaraokeJSON(karaokeJSON: KaraokeJSON): string {
  return JSON.stringify(karaokeJSON, null, 2);
}

/**
 * Скачивает JSON файл
 */
export function downloadKaraokeJSON(karaokeJSON: KaraokeJSON, filename: string = 'karaoke.json'): void {
  const jsonString = exportKaraokeJSON(karaokeJSON);
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  
  setTimeout(() => URL.revokeObjectURL(url), 100);
}
