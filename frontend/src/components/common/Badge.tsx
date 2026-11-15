import React from 'react';

export type BadgeVariant =
  | 'positive'
  | 'negative'
  | 'neutral'
  | 'indigo'
  | 'amber'
  | 'slate'
  | 'emerald'
  | 'rose'
  | 'zinc';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  size?: 'sm' | 'md';
  className?: string;
  title?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'zinc',
  size = 'sm',
  className = '',
  title,
}) => {
  const variantStyles: Record<BadgeVariant, string> = {
    positive: 'bg-emerald-500/[0.08] text-emerald-300 border-emerald-500/30 shadow-[inset_0_1px_0_0_rgba(16,185,129,0.15)]',
    negative: 'bg-rose-500/[0.08] text-rose-300 border-rose-500/30 shadow-[inset_0_1px_0_0_rgba(244,63,94,0.15)]',
    neutral: 'bg-amber-500/[0.08] text-amber-300 border-amber-500/30 shadow-[inset_0_1px_0_0_rgba(245,158,11,0.15)]',
    indigo: 'bg-white/[0.05] text-zinc-200 border-white/[0.1] shadow-[inset_0_1px_0_0_rgba(255,255,255,0.06)]',
    amber: 'bg-amber-500/[0.08] text-amber-300 border-amber-500/30 shadow-[inset_0_1px_0_0_rgba(245,158,11,0.15)]',
    slate: 'bg-white/[0.03] text-zinc-400 border-white/[0.06]',
    emerald: 'bg-emerald-500/[0.08] text-emerald-300 border-emerald-500/30 shadow-[inset_0_1px_0_0_rgba(16,185,129,0.15)]',
    rose: 'bg-rose-500/[0.08] text-rose-300 border-rose-500/30 shadow-[inset_0_1px_0_0_rgba(244,63,94,0.15)]',
    zinc: 'bg-white/[0.05] text-zinc-300 border-white/[0.1] shadow-[inset_0_1px_0_0_rgba(255,255,255,0.06)]',
  };

  const sizeStyles = size === 'sm' ? 'px-2 py-0.5 text-[10px]' : 'px-2.5 py-1 text-xs';

  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 font-mono tabular-nums font-medium rounded border tracking-wider uppercase ${variantStyles[variant]} ${sizeStyles} ${className}`}
    >
      {children}
    </span>
  );
};
