/**
 * Auto Lyrics Generator - автоматическая генерация lyrics.json из MIDI
 */
import { KaraokeJSON } from './karaokeJSON';
import { LyricsFile, LyricSegment } from './lyricsAligner';

export interface AutoLyricsOptions {
  placeholderText?: string; // текст-заглушка для каждой ноты (по умолчанию "la")
  phraseGap?: number; // пауза между фразами в секундах (по умолчанию 0.5)
  useSyllables?: boolean; // использовать слоги вместо целых фраз (по умолчанию false)
}

const DEFAULT_OPTIONS: AutoLyricsOptions = {
  placeholderText: 'la',
  phraseGap: 0.5,
  useSyllables: false,
};

/**
 * Автоматически генерирует lyrics.json из MIDI
 */
export function generateAutoLyrics(
  karaokeJSON: KaraokeJSON,
  options: AutoLyricsOptions = {}
): LyricsFile {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  console.log('[AutoLyrics] Generating lyrics from MIDI...');
  
  const phrases = karaokeJSON.phrases;
  
  if (phrases.length === 0) {
    console.log('[AutoLyrics] No phrases found');
    return {
      title: 'Auto-generated',
      lyrics: [],
    };
  }
  
  const lyrics: LyricSegment[] = [];
  
  if (opts.useSyllables) {
    // Режим слогов: каждая нота = один слог
    for (const phrase of phrases) {
      for (const note of phrase.notes) {
        lyrics.push({
          start: note.start,
          end: note.end,
          text: opts.placeholderText!,
        });
      }
    }
  } else {
    // Режим фраз: каждая фраза = одна строка текста
    for (const phrase of phrases) {
      // Считаем количество нот в фразе
      const noteCount = phrase.notes.length;
      
      // Создаём текст-заглушку (повторяем placeholder нужное количество раз)
      const text = Array(noteCount).fill(opts.placeholderText).join('-');
      
      lyrics.push({
        start: phrase.start,
        end: phrase.end,
        text: text,
      });
    }
  }
  
  console.log(`[AutoLyrics] Generated ${lyrics.length} segments`);
  
  return {
    title: 'Auto-generated',
    lyrics,
  };
}

/**
 * Экспортирует автоматически сгенерированный lyrics.json
 */
export function exportAutoLyrics(lyrics: LyricsFile): string {
  return JSON.stringify(lyrics, null, 2);
}

/**
 * Скачивает автоматически сгенерированный lyrics.json
 */
export function downloadAutoLyrics(lyrics: LyricsFile, filename: string = 'lyrics_auto.json'): void {
  const jsonString = exportAutoLyrics(lyrics);
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
