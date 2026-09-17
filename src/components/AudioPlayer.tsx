import { useRef, useState, useEffect, useCallback } from 'react';

interface AudioPlayerProps {
  audioUrl: string;
  label: string;
  color: 'violet' | 'pink' | 'blue' | 'amber' | 'emerald';
}

const COLOR_MAP = {
  violet: { bg: 'bg-violet-500', text: 'text-violet-400', light: 'bg-violet-500/20' },
  pink: { bg: 'bg-pink-500', text: 'text-pink-400', light: 'bg-pink-500/20' },
  blue: { bg: 'bg-blue-500', text: 'text-blue-400', light: 'bg-blue-500/20' },
  amber: { bg: 'bg-amber-500', text: 'text-amber-400', light: 'bg-amber-500/20' },
  emerald: { bg: 'bg-emerald-500', text: 'text-emerald-400', light: 'bg-emerald-500/20' },
};

export function AudioPlayer({ audioUrl, label, color }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const colors = COLOR_MAP[color];

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleLoadedMetadata = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioUrl]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;
    
    const time = parseFloat(e.target.value);
    audio.currentTime = time;
    setCurrentTime(time);
  }, []);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="flex items-center gap-3">
      <audio ref={audioRef} src={audioUrl} preload="metadata" />
      
      {/* Play button */}
      <button
        onClick={togglePlay}
        className={`w-10 h-10 rounded-full ${colors.light} ${colors.text} flex items-center justify-center transition-all hover:scale-110`}
      >
        {isPlaying ? (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <rect x="3" y="2" width="4" height="12" rx="1" />
            <rect x="9" y="2" width="4" height="12" rx="1" />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M4 2l10 6-10 6V2z" />
          </svg>
        )}
      </button>

      {/* Progress bar */}
      <div className="flex-1 flex items-center gap-2">
        <span className="text-xs text-gray-400 w-10 text-right font-mono">{formatTime(currentTime)}</span>
        <div className="flex-1 relative h-2 bg-white/10 rounded-full overflow-hidden">
          <div 
            className={`absolute top-0 left-0 h-full ${colors.bg} rounded-full transition-all duration-100`}
            style={{ width: `${progress}%` }}
          />
          <input
            type="range"
            min="0"
            max={duration || 0}
            step="0.1"
            value={currentTime}
            onChange={handleSeek}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
        </div>
        <span className="text-xs text-gray-400 w-10 font-mono">{formatTime(duration)}</span>
      </div>

      {/* Label */}
      <span className={`text-xs font-medium ${colors.text} hidden sm:inline`}>{label}</span>
    </div>
  );
}
