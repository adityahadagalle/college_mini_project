import React from 'react';
import { Cpu, ShieldCheck, User as UserIcon, Menu, X, ExternalLink, Activity } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

interface NavbarProps {
  onOpenProfile?: () => void;
  mobileMenuOpen?: boolean;
  onToggleMobileMenu?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  onOpenProfile,
  mobileMenuOpen = false,
  onToggleMobileMenu,
}) => {
  const { user } = useAuth();

  return (
    <header className="h-[52px] border-b border-white/[0.08] bg-[#090A0F]/80 backdrop-blur-xl px-4 sm:px-5 flex items-center justify-between sticky top-0 z-40 shadow-[0_4px_24px_-4px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)]">
      <div className="flex items-center gap-3">
        {/* Mobile Hamburger Toggle */}
        <button
          onClick={onToggleMobileMenu}
          aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={mobileMenuOpen}
          className="md:hidden p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.06] transition-colors"
        >
          {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
        </button>

        {/* Brand Logo & Name */}
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-zinc-700 via-zinc-800 to-zinc-950 border border-white/[0.15] flex items-center justify-center text-white shadow-[inset_0_1px_0_0_rgba(255,255,255,0.2)]">
            <Cpu className="w-4 h-4 text-zinc-100" />
          </div>
          <span className="font-bold text-white text-sm tracking-tight flex items-center gap-1">
            Sentiment<span className="text-zinc-400 font-light">AI</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.2 rounded bg-white/[0.06] text-zinc-300 ml-1 border border-white/[0.1] font-normal tracking-wider">
              Enterprise
            </span>
          </span>
        </div>

        {/* Multi-Engine Live Status Pill */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/[0.08] border border-emerald-500/25 text-[11px] text-emerald-300 font-mono shadow-[0_0_15px_rgba(16,185,129,0.12)]">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span>4 Engines Ready • Sentiment • ABSA • Emotion • Topics</span>
        </div>
      </div>

      <div className="flex items-center gap-2.5">
        <a
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono text-zinc-400 hover:text-zinc-100 bg-white/[0.02] hover:bg-white/[0.06] border border-white/[0.06] hover:border-white/[0.12] transition-all"
        >
          <span>Swagger Docs</span>
          <ExternalLink className="w-3 h-3 text-zinc-500" />
        </a>

        {user ? (
          <button
            onClick={onOpenProfile}
            aria-label="Open User Profile"
            className="flex items-center gap-2 bg-white/[0.03] hover:bg-white/[0.08] border border-white/[0.08] hover:border-white/[0.16] rounded-lg px-2.5 py-1 text-left transition-all active:scale-[0.98] shadow-xs"
          >
            <div className="w-6 h-6 rounded bg-gradient-to-br from-zinc-700 to-zinc-900 text-white flex items-center justify-center font-bold text-xs border border-white/[0.15] shadow-inner">
              {user.full_name ? user.full_name[0].toUpperCase() : <UserIcon className="w-3.5 h-3.5" />}
            </div>
            <div className="hidden sm:block text-xs">
              <p className="font-semibold text-zinc-100 leading-none">{user.full_name || 'Enterprise User'}</p>
              <p className="text-[10px] text-zinc-400 mt-0.5 font-mono">{user.email}</p>
            </div>
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-300 flex items-center gap-1.5 bg-white/[0.04] px-2.5 py-1 rounded-lg border border-white/[0.08] font-mono">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> OAuth2 / JWT
            </span>
          </div>
        )}
      </div>
    </header>
  );
};
