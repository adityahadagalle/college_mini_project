import React from 'react';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'card' | 'table' | 'chart';
  count?: number;
}

export const SkeletonLoader: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'text',
  count = 1,
}) => {
  const renderItem = (index: number) => {
    if (variant === 'card') {
      return (
        <div
          key={index}
          className={`p-5 rounded-xl bg-[#0F1115] border border-zinc-800/80 animate-pulse space-y-3 ${className}`}
        >
          <div className="h-3.5 w-1/3 bg-zinc-800/80 rounded" />
          <div className="h-7 w-1/2 bg-zinc-800/60 rounded" />
          <div className="h-3 w-2/3 bg-zinc-800/40 rounded" />
        </div>
      );
    }

    if (variant === 'chart') {
      return (
        <div
          key={index}
          className={`p-5 rounded-xl bg-[#0F1115] border border-zinc-800/80 animate-pulse h-64 flex items-end gap-3 ${className}`}
        >
          {[40, 75, 55, 90, 60, 80, 45].map((h, i) => (
            <div
              key={i}
              className="flex-1 bg-zinc-800/60 rounded-t"
              style={{ height: `${h}%` }}
            />
          ))}
        </div>
      );
    }

    if (variant === 'table') {
      return (
        <div key={index} className="space-y-2 animate-pulse">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="h-10 w-full bg-[#0F1115] border border-zinc-800/60 rounded-lg"
            />
          ))}
        </div>
      );
    }

    return (
      <div
        key={index}
        className={`h-4 bg-zinc-800/60 rounded animate-pulse w-full ${className}`}
      />
    );
  };

  return (
    <div className="space-y-3">
      {Array.from({ length: count }).map((_, idx) => renderItem(idx))}
    </div>
  );
};
