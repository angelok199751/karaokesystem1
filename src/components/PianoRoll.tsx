import { useEffect, useRef, useState } from 'react';
import { KaraokeNote } from '../utils/transcription/karaokeJSON';

interface PianoRollProps {
  notes: KaraokeNote[];
  selectedNoteIndex: number | null;
  onNoteClick: (index: number) => void;
  currentTime?: number;
  onTimeChange?: (time: number) => void;
}

export function PianoRoll({ 
  notes, 
  selectedNoteIndex, 
  onNoteClick,
  currentTime = 0,
  onTimeChange 
}: PianoRollProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Настройки отображения
  const PIXELS_PER_SECOND = 100;
  const PIXELS_PER_NOTE = 8;
  const MIN_NOTE = 48; // C3
  const MAX_NOTE = 84; // C6
  const NOTE_RANGE = MAX_NOTE - MIN_NOTE;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Вычисляем размеры
    const totalTime = notes.length > 0 
      ? Math.max(...notes.map(n => n.end)) 
      : 10;
    
    const width = Math.max(totalTime * PIXELS_PER_SECOND, canvas.parentElement?.clientWidth || 800);
    const height = NOTE_RANGE * PIXELS_PER_NOTE;

    canvas.width = width;
    canvas.height = height;

    // Очищаем canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);

    // Рисуем сетку
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;

    // Горизонтальные линии (ноты)
    for (let i = 0; i <= NOTE_RANGE; i++) {
      const y = i * PIXELS_PER_NOTE;
      const noteNum = MAX_NOTE - i;
      
      // Подсветка октав
      if (noteNum % 12 === 0) {
        ctx.strokeStyle = '#3a3a3a';
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = '#2a2a2a';
        ctx.lineWidth = 1;
      }
      
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Вертикальные линии (такты)
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;
    for (let t = 0; t <= totalTime; t += 1) {
      const x = t * PIXELS_PER_SECOND;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Рисуем ноты
    notes.forEach((note, index) => {
      const x = note.start * PIXELS_PER_SECOND;
      const y = (MAX_NOTE - note.note) * PIXELS_PER_NOTE;
      const noteWidth = (note.end - note.start) * PIXELS_PER_SECOND;
      const noteHeight = PIXELS_PER_NOTE;

      // Цвет ноты
      if (index === selectedNoteIndex) {
        ctx.fillStyle = '#fbbf24'; // Жёлтый для выбранной
      } else {
        ctx.fillStyle = '#8b5cf6'; // Фиолетовый для остальных
      }

      ctx.fillRect(x, y, noteWidth, noteHeight);

      // Обводка
      ctx.strokeStyle = index === selectedNoteIndex ? '#f59e0b' : '#7c3aed';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, noteWidth, noteHeight);
    });

    // Рисуем текущую позицию воспроизведения
    if (currentTime > 0) {
      const x = currentTime * PIXELS_PER_SECOND;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

  }, [notes, selectedNoteIndex, currentTime]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left + scrollLeft);
    const y = (e.clientY - rect.top + scrollTop);

    // Проверяем клик по ноте
    for (let i = 0; i < notes.length; i++) {
      const note = notes[i];
      const noteX = note.start * PIXELS_PER_SECOND;
      const noteY = (MAX_NOTE - note.note) * PIXELS_PER_NOTE;
      const noteWidth = (note.end - note.start) * PIXELS_PER_SECOND;
      const noteHeight = PIXELS_PER_NOTE;

      if (x >= noteX && x <= noteX + noteWidth &&
          y >= noteY && y <= noteY + noteHeight) {
        onNoteClick(i);
        return;
      }
    }

    // Клик по пустому месту - перемещаем позицию воспроизведения
    if (onTimeChange) {
      const time = x / PIXELS_PER_SECOND;
      onTimeChange(time);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button === 1 || e.button === 2) { // Средняя или правая кнопка
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
      e.preventDefault();
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging) return;

    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;

    setScrollLeft(prev => Math.max(0, prev - dx));
    setScrollTop(prev => Math.max(0, prev - dy));
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div 
      ref={containerRef}
      className="relative w-full h-96 bg-gray-900 rounded-lg overflow-hidden border border-gray-700"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onContextMenu={(e) => e.preventDefault()}
    >
      <div 
        ref={(el) => {
          if (el) {
            el.scrollLeft = scrollLeft;
            el.scrollTop = scrollTop;
          }
        }}
        className="absolute inset-0 overflow-auto"
        onScroll={(e) => {
          setScrollLeft(e.currentTarget.scrollLeft);
          setScrollTop(e.currentTarget.scrollTop);
        }}
      >
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="cursor-pointer"
        />
      </div>
      
      {/* Легенда */}
      <div className="absolute bottom-2 right-2 bg-black/70 px-3 py-1 rounded text-xs text-gray-300">
        Клик: выбрать ноту | ПКМ+перетаскивание: прокрутка
      </div>
    </div>
  );
}
