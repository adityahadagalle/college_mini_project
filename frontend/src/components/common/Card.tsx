import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hoverable?: boolean;
}

export const Card: React.FC<CardProps> = ({ children, className = '', hoverable = false }) => {
  return (
    <div
      className={`relative rounded-xl p-5 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.5),inset_0_1px_0_0_rgba(255,255,255,0.06)] backdrop-blur-md transition-all duration-150 ${
        hoverable
          ? 'hover:border-white/[0.18] hover:shadow-[0_8px_32px_-4px_rgba(0,0,0,0.7),inset_0_1px_0_0_rgba(255,255,255,0.1)] hover:-translate-y-[1px]'
          : ''
      } ${className}`}
    >
      {children}
    </div>
  );
};
