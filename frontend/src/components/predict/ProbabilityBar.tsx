import React from 'react';

interface ProbabilityBarProps {
  label: string;
  percentage: number; // 0.0 – 1.0
  colorClass: string;
}

export const ProbabilityBar: React.FC<ProbabilityBarProps> = ({
  label,
  percentage,
  colorClass,
}) => {
  const safePercent = isNaN(percentage) || !isFinite(percentage) ? 0 : Math.max(0, Math.min(percentage, 1));
  const displayValue = (safePercent * 100).toFixed(1);
  const barWidth = Math.max(safePercent * 100, safePercent > 0 ? 2 : 0);

  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center text-xs">
        <span className="text-zinc-400 font-medium">{label}</span>
        <span className="text-zinc-200 font-mono font-semibold tabular-nums">
          {displayValue}%
        </span>
      </div>
      <div className="h-2 w-full bg-zinc-950 rounded-full overflow-hidden border border-zinc-800">
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${colorClass}`}
          style={{ width: `${barWidth}%` }}
          role="progressbar"
          aria-valuenow={safePercent * 100}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label}: ${displayValue}%`}
        />
      </div>
    </div>
  );
};
