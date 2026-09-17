/**
 * Lyrics Aligner - привязка нот к тексту
 */
import { KaraokeNote, KaraokeJSON } from './karaokeJSON';

export interface LyricSegment {
  start: number;
  end: number;
  text: string;
}

export interface LyricsFile {
  title: string;
  lyrics: LyricSegment[];
}

export interface AlignedNote {
  start: number;
  end: number;
  pitch: number;
  text: string;
}

export interface AlignedSegment {
  start: number;
  end: number;
  text: string;
  notes: AlignedNote[];
}

export interface AlignedKaraokeFile {
  title: string;
  bpm: number;
  segments: AlignedSegment[];
}

/**
 * Привязывает ноты к тексту
 */
export function alignNotesToLyrics(
  karaokeJSON: KaraokeJSON,
  lyrics: LyricsFile
): AlignedKaraokeFile {
  console.log('[LyricsAligner] Aligning notes to lyrics...');
  
  // Получаем все ноты
  const allNotes: KaraokeNote[] = karaokeJSON.phrases.flatMap(p => p.notes);
  
  if (allNotes.length === 0) {
    console.log('[LyricsAligner] No notes to align');
    return {
      title: lyrics.title,
      bpm: karaokeJSON.bpm,
      segments: lyrics.lyrics.map(seg => ({
        start: seg.start,
        end: seg.end,
        text: seg.text,
        notes: [],
      })),
    };
  }
  
  console.log(`[LyricsAligner] Aligning ${allNotes.length} notes to ${lyrics.lyrics.length} segments`);
  
  // Для каждого сегмента текста находим ноты в этом временном окне
  const segments: AlignedSegment[] = lyrics.lyrics.map(segment => {
    // Находим ноты, которые попадают в этот сегмент
    const segmentNotes = allNotes.filter(note => {
      // Нота должна начинаться или заканчиваться в пределах сегмента
      return (note.start >= segment.start && note.start < segment.end) ||
             (note.end > segment.start && note.end <= segment.end);
    });
    
    // Распределяем ноты по символам текста
    const alignedNotes = distributeNotesToText(segmentNotes, segment.text, segment.start, segment.end);
    
    return {
      start: segment.start,
      end: segment.end,
      text: segment.text,
      notes: alignedNotes,
    };
  });
  
  console.log(`[LyricsAligner] Aligned ${segments.reduce((sum, s) => sum + s.notes.length, 0)} notes`);
  
  return {
    title: lyrics.title,
    bpm: karaokeJSON.bpm,
    segments,
  };
}

/**
 * Распределяет ноты по символам текста
 */
function distributeNotesToText(
  notes: KaraokeNote[],
  text: string,
  segmentStart: number,
  segmentEnd: number
): AlignedNote[] {
  if (notes.length === 0) return [];
  
  const segmentDuration = segmentEnd - segmentStart;
  const charCount = text.length;
  
  if (charCount === 0) return [];
  
  // Сортируем ноты по времени
  const sortedNotes = [...notes].sort((a, b) => a.start - b.start);
  
  // Простая стратегия: равномерно распределяем ноты по символам
  // Каждая нота получает пропорциональную часть текста
  const alignedNotes: AlignedNote[] = [];
  
  // Если нот меньше, чем символов - распределяем равномерно
  if (sortedNotes.length <= charCount) {
    const charsPerNote = Math.floor(charCount / sortedNotes.length);
    
    for (let i = 0; i < sortedNotes.length; i++) {
      const note = sortedNotes[i];
      const startChar = i * charsPerNote;
      const endChar = i === sortedNotes.length - 1 
        ? charCount 
        : (i + 1) * charsPerNote;
      
      const noteText = text.substring(startChar, endChar);
      
      alignedNotes.push({
        start: note.start,
        end: note.end,
        pitch: note.note,
        text: noteText,
      });
    }
  } else {
    // Если нот больше, чем символов - группируем ноты по символам
    const notesPerChar = Math.ceil(sortedNotes.length / charCount);
    
    let noteIndex = 0;
    for (let charIdx = 0; charIdx < charCount && noteIndex < sortedNotes.length; charIdx++) {
      const char = text[charIdx];
      const groupEnd = Math.min(noteIndex + notesPerChar, sortedNotes.length);
      
      // Берём первую ноту из группы как основную
      const mainNote = sortedNotes[noteIndex];
      
      alignedNotes.push({
        start: mainNote.start,
        end: sortedNotes[groupEnd - 1].end, // конец последней ноты в группе
        pitch: mainNote.note,
        text: char,
      });
      
      noteIndex = groupEnd;
    }
  }
  
  return alignedNotes;
}

/**
 * Экспортирует выровненный файл в JSON
 */
export function exportAlignedKaraoke(aligned: AlignedKaraokeFile): string {
  return JSON.stringify(aligned, null, 2);
}

/**
 * Скачивает выровненный файл
 */
export function downloadAlignedKaraoke(aligned: AlignedKaraokeFile, filename: string = 'karaoke_aligned.json'): void {
  const jsonString = exportAlignedKaraoke(aligned);
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
