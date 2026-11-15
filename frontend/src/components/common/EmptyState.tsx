import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  actionLabel,
  onAction,
}) => {
  return (
    <div className="p-10 text-center bg-gradient-to-b from-[#141720]/80 to-[#0D0F14]/90 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl flex flex-col items-center justify-center space-y-3.5 backdrop-blur-md">
      <div className="w-12 h-12 rounded-xl bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-400">
        {icon || <Inbox className="w-5 h-5 text-zinc-400" />}
      </div>
      <div>
        <h4 className="text-sm font-semibold text-zinc-200">{title}</h4>
        <p className="text-xs text-zinc-400 mt-1 max-w-sm mx-auto leading-relaxed font-sans">{description}</p>
      </div>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="mt-2 px-3.5 py-1.5 bg-gradient-to-b from-white via-zinc-100 to-zinc-200 hover:to-white text-zinc-950 rounded-lg text-xs font-semibold shadow-[0_1px_2px_rgba(0,0,0,0.4),0_0_12px_rgba(255,255,255,0.15),inset_0_1px_0_rgba(255,255,255,1)] transition-all active:scale-[0.98]"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
};
