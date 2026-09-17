import { useCallback, useRef, useState } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
}

const ACCEPTED_TYPES = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac', 'audio/mp4', 'audio/aac', 'audio/webm'];
const ACCEPTED_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac', '.webm'];

export function FileUpload({ onFileSelect }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    const isValidType = ACCEPTED_TYPES.includes(file.type) || 
      ACCEPTED_EXTENSIONS.some(ext => file.name.toLowerCase().endsWith(ext));
    
    if (!isValidType) {
      alert('Please select a valid audio file (MP3, WAV, OGG, FLAC, M4A, AAC, WebM)');
      return;
    }
    
    onFileSelect(file);
  }, [onFileSelect]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={() => inputRef.current?.click()}
      className={`
        relative cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all
        ${isDragging 
          ? 'border-violet-400 bg-violet-500/10 scale-[1.02]' 
          : 'border-white/20 hover:border-violet-400/50 hover:bg-white/5'
        }
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        onChange={handleInputChange}
        className="hidden"
      />
      
      <div className="space-y-4">
        <div className="text-5xl">
          {isDragging ? '📂' : '🎵'}
        </div>
        <div>
          <p className="text-lg font-medium text-white">
            {isDragging ? 'Drop your audio file here' : 'Drop audio file or click to browse'}
          </p>
          <p className="text-sm text-gray-400 mt-1">
            Supports MP3, WAV, OGG, FLAC, M4A, AAC, WebM
          </p>
        </div>
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-500/20 text-violet-300 text-sm font-medium border border-violet-500/30">
          <span>📁</span>
          <span>Choose File</span>
        </div>
      </div>
    </div>
  );
}
