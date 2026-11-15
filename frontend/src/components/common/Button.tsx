import React from 'react';
import { RefreshCw } from 'lucide-react';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: ButtonVariant;
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ElementType;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  icon: Icon,
  disabled,
  className = '',
  ...props
}) => {
  const variantStyles: Record<ButtonVariant, string> = {
    primary:
      'bg-gradient-to-b from-white via-zinc-100 to-zinc-200 hover:to-white text-zinc-950 font-semibold border border-white/60 shadow-[0_1px_2px_rgba(0,0,0,0.4),0_0_12px_rgba(255,255,255,0.15),inset_0_1px_0_rgba(255,255,255,1)] hover:shadow-[0_2px_10px_rgba(255,255,255,0.25)] active:scale-[0.98]',
    secondary:
      'bg-[#141720]/90 hover:bg-[#1A1E2A] text-zinc-200 border border-white/[0.08] hover:border-white/[0.18] shadow-[0_1px_2px_rgba(0,0,0,0.3),inset_0_1px_0_0_rgba(255,255,255,0.06)] active:scale-[0.98]',
    ghost:
      'bg-transparent hover:bg-white/[0.06] text-zinc-400 hover:text-zinc-100 border-transparent active:scale-[0.98]',
    danger:
      'bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 shadow-[0_1px_2px_rgba(244,63,94,0.1),inset_0_1px_0_0_rgba(255,255,255,0.05)] active:scale-[0.98]',
  };

  const sizeStyles = {
    sm: 'px-2.5 py-1.5 text-xs rounded-lg',
    md: 'px-3.5 py-2 text-xs font-semibold rounded-lg',
    lg: 'px-4.5 py-2.5 text-sm font-semibold rounded-lg',
  };

  return (
    <button
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center gap-2 font-medium border transition-all duration-140 disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-950 cursor-pointer ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {loading ? (
        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
      ) : (
        Icon && <Icon className="w-3.5 h-3.5 shrink-0" />
      )}
      <span>{children}</span>
    </button>
  );
};
