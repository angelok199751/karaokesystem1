/**
 * MIDI Cleaner - очистка MIDI от ложных и мелких нот
 */
import { KaraokeNote, KaraokeJSON } from './karaokeJSON';

export interface CleanerOptions {
  minNoteDuration?: number; // минимальная длительность ноты в секундах (по умолчанию 0.08)
  maxNoteDuration?: number; // максимальная длительность ноты в секундах (по умолчанию 5.0)
  mergeLegato?: boolean; // сливать соседние ноты одной высоты (по умолчанию true)
  legatoThreshold?: number; // порог для слияния в секундах (по умолчанию 0.05)
  minGapBetweenNotes?: number; // минимальный промежуток между нотами (по умолчанию 0.02)
}

const DEFAULT_OPTIONS: CleanerOptions = {
  minNoteDuration: 0.08,
  maxNoteDuration: 5.0,
  mergeLegato: true,
  legatoThreshold: 0.05,
  minGapBetweenNotes: 0.02,
};

/**
 * Очищает MIDI от ложных и мелких нот
 */
export function cleanMidi(
  karaokeJSON: KaraokeJSON,
  options: CleanerOptions = {}
): KaraokeJSON {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  console.log('[MidiCleaner] Cleaning with options:', opts);
  
  // Получаем все ноты
  const allNotes: KaraokeNote[] = karaokeJSON.phrases.flatMap(p => p.notes);
  
  if (allNotes.length === 0) {
    console.log('[MidiCleaner] No notes to clean');
    return karaokeJSON;
  }
  
  console.log(`[MidiCleaner] Starting with ${allNotes.length} notes`);
  
  // Шаг 1: Удаляем слишком короткие ноты
  let cleanedNotes = removeShortNotes(allNotes, opts.minNoteDuration!);
  console.log(`[MidiCleaner] After removing short notes: ${cleanedNotes.length} notes`);
  
  // Шаг 2: Удаляем слишком длинные ноты
  cleanedNotes = removeLongNotes(cleanedNotes, opts.maxNoteDuration!);
  console.log(`[MidiCleaner] After removing long notes: ${cleanedNotes.length} notes`);
  
  // Шаг 3: Сливаем соседние ноты одной высоты (легато)
  if (opts.mergeLegato) {
    cleanedNotes = mergeLegatoNotes(cleanedNotes, opts.legatoThreshold!);
    console.log(`[MidiCleaner] After merging legato: ${cleanedNotes.length} notes`);
  }
  
  // Шаг 4: Убираем слишком близкие ноты
  cleanedNotes = removeOverlappingNotes(cleanedNotes, opts.minGapBetweenNotes!);
  console.log(`[MidiCleaner] After removing overlaps: ${cleanedNotes.length} notes`);
  
  // Шаг 5: Сортируем по времени
  cleanedNotes.sort((a, b) => a.start - b.start);
  
  // Группируем обратно в фразы
  const phrases = groupNotesIntoPhrases(cleanedNotes);
  
  console.log(`[MidiCleaner] Final result: ${cleanedNotes.length} notes in ${phrases.length} phrases`);
  
  return {
    ...karaokeJSON,
    phrases,
  };
}

/**
 * Удаляет ноты короче minDuration
 */
function removeShortNotes(notes: KaraokeNote[], minDuration: number): KaraokeNote[] {
  return notes.filter(note => {
    const duration = note.end - note.start;
    return duration >= minDuration;
  });
}

/**
 * Удаляет ноты длиннее maxDuration
 */
function removeLongNotes(notes: KaraokeNote[], maxDuration: number): KaraokeNote[] {
  return notes.filter(note => {
    const duration = note.end - note.start;
    return duration <= maxDuration;
  });
}

/**
 * Сливает соседние ноты одной высоты (легато)
 */
function mergeLegatoNotes(notes: KaraokeNote[], threshold: number): KaraokeNote[] {
  if (notes.length === 0) return [];
  
  // Сортируем по времени начала
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  const merged: KaraokeNote[] = [sorted[0]];
  
  for (let i = 1; i < sorted.length; i++) {
    const current = sorted[i];
    const previous = merged[merged.length - 1];
    
    // Проверяем, можно ли слить
    const gap = current.start - previous.end;
    const samePitch = current.note === previous.note;
    
    if (samePitch && gap <= threshold) {
      // Сливаем: расширяем предыдущую ноту
      previous.end = Math.max(previous.end, current.end);
    } else {
      // Добавляем как новую
      merged.push(current);
    }
  }
  
  return merged;
}

/**
 * Удаляет перекрывающиеся ноты (оставляет более длинную)
 */
function removeOverlappingNotes(notes: KaraokeNote[], minGap: number): KaraokeNote[] {
  if (notes.length === 0) return [];
  
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  const result: KaraokeNote[] = [sorted[0]];
  
  for (let i = 1; i < sorted.length; i++) {
    const current = sorted[i];
    const previous = result[result.length - 1];
    
    // Проверяем перекрытие
    const gap = current.start - previous.end;
    
    if (gap < minGap) {
      // Перекрываются - оставляем более длинную
      const currentDuration = current.end - current.start;
      const previousDuration = previous.end - previous.start;
      
      if (currentDuration > previousDuration) {
        result[result.length - 1] = current;
      }
    } else {
      result.push(current);
    }
  }
  
  return result;
}

/**
 * Группирует ноты в фразы (на основе пауз)
 */
function groupNotesIntoPhrases(notes: KaraokeNote[]): KaraokeJSON['phrases'] {
  if (notes.length === 0) return [];
  
  const PHRASE_GAP = 0.5; // пауза между фразами в секундах
  const phrases: KaraokeJSON['phrases'] = [];
  let currentPhrase: KaraokeNote[] = [notes[0]];
  
  for (let i = 1; i < notes.length; i++) {
    const prev = notes[i - 1];
    const curr = notes[i];
    const gap = curr.start - prev.end;
    
    if (gap > PHRASE_GAP) {
      // Новая фраза
      phrases.push({
        notes: currentPhrase,
        start: currentPhrase[0].start,
        end: currentPhrase[currentPhrase.length - 1].end,
      });
      currentPhrase = [curr];
    } else {
      currentPhrase.push(curr);
    }
  }
  
  // Добавляем последнюю фразу
  if (currentPhrase.length > 0) {
    phrases.push({
      notes: currentPhrase,
      start: currentPhrase[0].start,
      end: currentPhrase[currentPhrase.length - 1].end,
    });
  }
  
  return phrases;
}
