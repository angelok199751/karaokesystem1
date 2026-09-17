import { useState, useRef, useEffect } from 'react';
import * as Tone from 'tone';
import { PianoRoll } from './PianoRoll';
import { KaraokeNote, KaraokeJSON } from '../utils/transcription/karaokeJSON';
import { MidiCleanerPanel } from './MidiCleanerPanel';
import { LyricsImporter } from './LyricsImporter';
import { cleanMidi, CleanerOptions } from '../utils/transcription/midiCleaner';
import { alignNotesToLyrics, downloadAlignedKaraoke, LyricsFile, AlignedKaraokeFile } from '../utils/transcription/lyricsAligner';

interface MidiEditorProps {
  karaokeJSON: KaraokeJSON;
  onUpdate: (updated: KaraokeJSON) => void;
}

export function MidiEditor({ karaokeJSON, onUpdate }: MidiEditorProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedNoteIndex, setSelectedNoteIndex] = useState<number | null>(null);
  const [volume, setVolume] = useState(-10);
  const [lyrics, setLyrics] = useState<LyricsFile | null>(null);
  const [alignedData, setAlignedData] = useState<AlignedKaraokeFile | null>(null);
  const synthRef = useRef<Tone.PolySynth | null>(null);
  const animationRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  // Все ноты из всех фраз
  const allNotes: KaraokeNote[] = karaokeJSON.phrases.flatMap(p => p.notes);

  // Инициализация синтезатора
  useEffect(() => {
    synthRef.current = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: 'triangle' },
      envelope: { attack: 0.02, decay: 0.1, sustain: 0.3, release: 0.5 }
    }).toDestination();
    
    Tone.Destination.volume.value = volume;

    return () => {
      synthRef.current?.dispose();
    };
  }, []);

  // Обновление громкости
  useEffect(() => {
    if (synthRef.current) {
      Tone.Destination.volume.value = volume;
    }
  }, [volume]);

  // Воспроизведение MIDI
  const play = async () => {
    await Tone.start();
    
    if (!synthRef.current) return;

    setIsPlaying(true);
    startTimeRef.current = Tone.now() - currentTime;

    // Планируем все ноты
    allNotes.forEach(note => {
      if (note.start >= currentTime) {
        const duration = note.end - note.start;
        synthRef.current?.triggerAttackRelease(
          Tone.Frequency(note.note, 'midi').toNote(),
          duration,
          Tone.now() + (note.start - currentTime)
        );
      }
    });

    // Анимация текущей позиции
    const updateCurrentTime = () => {
      const elapsed = Tone.now() - startTimeRef.current;
      setCurrentTime(elapsed);

      const maxTime = allNotes.length > 0 ? Math.max(...allNotes.map(n => n.end)) : 0;
      if (elapsed < maxTime) {
        animationRef.current = requestAnimationFrame(updateCurrentTime);
      } else {
        stop();
      }
    };

    animationRef.current = requestAnimationFrame(updateCurrentTime);
  };

  const stop = () => {
    setIsPlaying(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    synthRef.current?.releaseAll();
  };

  const pause = () => {
    stop();
  };

  // Обновление ноты
  const updateNote = (index: number, updates: Partial<KaraokeNote>) => {
    const newPhrases = karaokeJSON.phrases.map(phrase => {
      const newNotes = phrase.notes.map((note, i) => {
        // Находим глобальный индекс
        const globalIndex = karaokeJSON.phrases.findIndex(p => p === phrase) * 1000 + i;
        if (globalIndex === index) {
          return { ...note, ...updates };
        }
        return note;
      });
      return { ...phrase, notes: newNotes };
    });

    onUpdate({ ...karaokeJSON, phrases: newPhrases });
  };

  // Удаление ноты
  const deleteNote = (index: number) => {
    const newPhrases = karaokeJSON.phrases.map(phrase => ({
      ...phrase,
      notes: phrase.notes.filter((_, i) => {
        const globalIndex = karaokeJSON.phrases.findIndex(p => p === phrase) * 1000 + i;
        return globalIndex !== index;
      })
    })).filter(phrase => phrase.notes.length > 0);

    onUpdate({ ...karaokeJSON, phrases: newPhrases });
    setSelectedNoteIndex(null);
  };

  // Добавление новой ноты
  const addNote = (time: number, note: number) => {
    const duration = 0.2; // Дефолтная длительность 200ms
    const newNote: KaraokeNote = {
      start: time,
      end: time + duration,
      note: note
    };

    // Добавляем в первую фразу или создаём новую
    if (karaokeJSON.phrases.length === 0) {
      const newPhrase = {
        notes: [newNote],
        start: time,
        end: time + duration
      };
      onUpdate({ ...karaokeJSON, phrases: [newPhrase] });
    } else {
      // Находим подходящую фразу (ближайшую по времени)
      let targetPhraseIndex = 0;
      let minDistance = Infinity;

      karaokeJSON.phrases.forEach((phrase, index) => {
        const distance = Math.abs(phrase.start - time);
        if (distance < minDistance) {
          minDistance = distance;
          targetPhraseIndex = index;
        }
      });

      const newPhrases = karaokeJSON.phrases.map((phrase, index) => {
        if (index === targetPhraseIndex) {
          const newNotes = [...phrase.notes, newNote].sort((a, b) => a.start - b.start);
          return {
            ...phrase,
            notes: newNotes,
            start: Math.min(phrase.start, time),
            end: Math.max(phrase.end, time + duration)
          };
        }
        return phrase;
      });

      onUpdate({ ...karaokeJSON, phrases: newPhrases });
    }
  };

  // Перетаскивание ноты
  const handleNoteDrag = (index: number, newStart: number, newNote: number) => {
    const note = allNotes[index % 1000];
    const duration = note.end - note.start;
    
    const newPhrases = karaokeJSON.phrases.map((phrase, phraseIndex) => {
      const newNotes = phrase.notes.map((n, noteIndex) => {
        const globalIndex = phraseIndex * 1000 + noteIndex;
        if (globalIndex === index) {
          return {
            start: newStart,
            end: newStart + duration,
            note: newNote
          };
        }
        return n;
      });
      
      // Обновляем границы фразы
      const phraseStart = newNotes.length > 0 ? Math.min(...newNotes.map(n => n.start)) : phrase.start;
      const phraseEnd = newNotes.length > 0 ? Math.max(...newNotes.map(n => n.end)) : phrase.end;
      
      return {
        ...phrase,
        notes: newNotes,
        start: phraseStart,
        end: phraseEnd
      };
    });

    onUpdate({ ...karaokeJSON, phrases: newPhrases });
  };

  // Получение глобального индекса ноты
  const getGlobalIndex = (phraseIndex: number, noteIndex: number): number => {
    return phraseIndex * 1000 + noteIndex;
  };

  // Обработка очистки MIDI
  const handleClean = (options: CleanerOptions) => {
    const cleaned = cleanMidi(karaokeJSON, options);
    onUpdate(cleaned);
  };

  // Обработка импорта текста
  const handleLyricsImport = (importedLyrics: LyricsFile) => {
    setLyrics(importedLyrics);
    
    // Автоматически выравниваем ноты по тексту
    const aligned = alignNotesToLyrics(karaokeJSON, importedLyrics);
    setAlignedData(aligned);
  };

  // Скачивание выровненного файла
  const handleDownloadAligned = () => {
    if (alignedData) {
      downloadAlignedKaraoke(alignedData, 'karaoke_aligned.json');
    }
  };

  const selectedNote = selectedNoteIndex !== null 
    ? allNotes[selectedNoteIndex % 1000]
    : null;

  return (
    <div className="space-y-4">
      {/* Контролы воспроизведения */}
      <div className="bg-gray-800 rounded-lg p-4 space-y-4">
        <div className="flex items-center gap-4">
          <button
            onClick={isPlaying ? pause : play}
            className="px-6 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium transition"
          >
            {isPlaying ? '⏸️ Пауза' : '▶️ Воспроизвести'}
          </button>
          
          <button
            onClick={stop}
            disabled={!isPlaying}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition disabled:opacity-50"
          >
            ⏹️ Стоп
          </button>

          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">🔊</span>
            <input
              type="range"
              min="-40"
              max="0"
              value={volume}
              onChange={(e) => setVolume(Number(e.target.value))}
              className="w-32"
            />
          </div>

          <div className="text-sm text-gray-400">
            {currentTime.toFixed(2)}s / {allNotes.length > 0 ? Math.max(...allNotes.map(n => n.end)).toFixed(2) : '0.00'}s
          </div>
        </div>

        {/* Прогресс-бар */}
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-purple-600 h-2 rounded-full transition-all"
            style={{
              width: `${allNotes.length > 0 ? (currentTime / Math.max(...allNotes.map(n => n.end))) * 100 : 0}%`
            }}
          />
        </div>
      </div>

      {/* Piano Roll */}
      <div>
        <h3 className="text-lg font-semibold mb-2">🎹 Piano Roll</h3>
        <PianoRoll
          notes={allNotes}
          selectedNoteIndex={selectedNoteIndex}
          onNoteClick={setSelectedNoteIndex}
          onNoteDrag={handleNoteDrag}
          onDoubleClick={addNote}
          currentTime={currentTime}
          onTimeChange={setCurrentTime}
        />
      </div>

      {/* Редактор выбранной ноты */}
      {selectedNote && selectedNoteIndex !== null && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">✏️ Редактирование ноты</h3>
            <button
              onClick={() => deleteNote(selectedNoteIndex)}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm transition"
            >
              🗑️ Удалить
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Нота (MIDI)</label>
              <input
                type="number"
                min="0"
                max="127"
                value={selectedNote.note}
                onChange={(e) => updateNote(selectedNoteIndex, { note: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
              />
              <div className="text-xs text-gray-500 mt-1">
                {Tone.Frequency(selectedNote.note, 'midi').toNote()}
              </div>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Начало (с)</label>
              <input
                type="number"
                step="0.001"
                min="0"
                value={selectedNote.start}
                onChange={(e) => updateNote(selectedNoteIndex, { start: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Конец (с)</label>
              <input
                type="number"
                step="0.001"
                min="0"
                value={selectedNote.end}
                onChange={(e) => updateNote(selectedNoteIndex, { end: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Длительность (с)</label>
              <div className="px-3 py-2 bg-gray-700 rounded border border-gray-600">
                {(selectedNote.end - selectedNote.start).toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MIDI Cleaner */}
      <MidiCleanerPanel onClean={handleClean} />

      {/* Lyrics Importer */}
      <LyricsImporter onImport={handleLyricsImport} />

      {/* Aligned Data Download */}
      {alignedData && (
        <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-green-300">✓ Текст выровнен</h3>
              <p className="text-sm text-green-200">
                {alignedData.segments.length} сегментов, {alignedData.segments.reduce((sum, s) => sum + s.notes.length, 0)} нот привязано
              </p>
            </div>
            <button
              onClick={handleDownloadAligned}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition"
            >
              ⬇️ Скачать выровненный файл
            </button>
          </div>
        </div>
      )}

      {/* Статистика */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-2">📊 Статистика</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-gray-400">Всего нот</div>
            <div className="text-xl font-bold">{allNotes.length}</div>
          </div>
          <div>
            <div className="text-gray-400">Фраз</div>
            <div className="text-xl font-bold">{karaokeJSON.phrases.length}</div>
          </div>
          <div>
            <div className="text-gray-400">BPM</div>
            <div className="text-xl font-bold">{karaokeJSON.bpm}</div>
          </div>
          <div>
            <div className="text-gray-400">Диапазон</div>
            <div className="text-xl font-bold">
              {allNotes.length > 0 
                ? `${Math.min(...allNotes.map(n => n.note))} - ${Math.max(...allNotes.map(n => n.note))}`
                : '-'
              }
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
