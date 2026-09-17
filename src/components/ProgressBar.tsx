interface ProgressBarProps {
  progress: number;
  message: string;
  variant: 'model' | 'separation';
}

export function ProgressBar({ progress, message, variant }: ProgressBarProps) {
  const colorClass = variant === 'model' 
    ? 'from-violet-500 to-indigo-500' 
    : 'from-emerald-500 to-teal-500';
  
  const bgColor = variant === 'model' ? 'bg-violet-500/20' : 'bg-emerald-500/20';
  const textColor = variant === 'model' ? 'text-violet-400' : 'text-emerald-400';

  return (
    <div className="p-4 rounded-xl bg-white/5 border border-white/10">
      <div className="flex items-center justify-between mb-2">
        <span className={`text-sm font-medium ${textColor}`}>
          {variant === 'model' ? '📦 Model' : '🎛️ Processing'}
        </span>
        <span className="text-sm font-mono text-gray-400">{Math.round(progress)}%</span>
      </div>
      
      <div className={`h-3 ${bgColor} rounded-full overflow-hidden`}>
        <div 
          className={`h-full bg-gradient-to-r ${colorClass} rounded-full transition-all duration-300 ease-out`}
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>
      
      <p className="text-xs text-gray-400 mt-2">{message}</p>
    </div>
  );
}
