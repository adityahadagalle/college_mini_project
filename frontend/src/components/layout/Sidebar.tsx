import React from 'react';
import {
  Brain,
  Layers,
  Zap,
  Split,
  Heart,
  Boxes,
  LayoutDashboard,
  TrendingUp,
  FileSpreadsheet,
  History,
  Award,
  Activity,
  Settings as SettingsIcon,
  Shield,
  Circle,
} from 'lucide-react';

interface SidebarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  mobileMenuOpen?: boolean;
  onCloseMobileMenu?: () => void;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  setCurrentTab,
  mobileMenuOpen = false,
  onCloseMobileMenu,
}) => {
  const sections: NavSection[] = [
    {
      title: 'INTELLIGENCE SUITE',
      items: [
        { id: 'intelligence', label: 'Review Intelligence', icon: Brain },
        { id: 'cross-feature', label: 'Cross-Feature Analytics', icon: Layers },
      ],
    },
    {
      title: 'SPECIALIZED ENGINES',
      items: [
        { id: 'predict', label: 'Single Prediction', icon: Zap },
        { id: 'absa', label: 'Aspect Explorer (ABSA)', icon: Split },
        { id: 'emotion', label: 'Emotion Intelligence', icon: Heart },
        { id: 'topics', label: 'Topic Discovery', icon: Boxes },
      ],
    },
    {
      title: 'PLATFORM & OPERATIONS',
      items: [
        { id: 'dashboard', label: 'Executive Dashboard', icon: LayoutDashboard },
        { id: 'analytics', label: 'Trends & Analytics', icon: TrendingUp },
        { id: 'batch', label: 'Batch Ingestion', icon: FileSpreadsheet },
        { id: 'history', label: 'Audit History', icon: History },
        { id: 'mlops', label: 'MLOps & Registry', icon: Award },
        { id: 'health', label: 'System Health', icon: Activity },
        { id: 'landing', label: 'Platform Architecture', icon: Shield },
        { id: 'settings', label: 'Settings', icon: SettingsIcon },
      ],
    },
  ];

  const handleTabClick = (id: string) => {
    setCurrentTab(id);
    if (onCloseMobileMenu) onCloseMobileMenu();
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileMenuOpen && (
        <div
          onClick={onCloseMobileMenu}
          aria-hidden="true"
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-30 md:hidden"
        />
      )}

      <aside
        className={`fixed md:static top-[52px] bottom-0 left-0 z-40 w-[236px] border-r border-white/[0.08] bg-[#090A0F]/85 backdrop-blur-xl p-3 flex flex-col justify-between shrink-0 overflow-y-auto transition-transform duration-200 md:translate-x-0 ${
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <nav className="space-y-4">
          {sections.map((section, sIdx) => (
            <div key={sIdx} className="space-y-1">
              <div className="px-2.5 py-1 text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.08em] font-mono">
                {section.title}
              </div>
              <div className="space-y-0.5">
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const isActive = currentTab === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => handleTabClick(item.id)}
                      className={`w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        isActive
                          ? 'bg-white/[0.08] text-white border border-white/[0.12] shadow-[0_2px_8px_rgba(0,0,0,0.5),inset_0_1px_0_0_rgba(255,255,255,0.08)] font-semibold'
                          : 'text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.04]'
                      }`}
                    >
                      <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-zinc-100' : 'text-zinc-500'}`} />
                      <span className="truncate">{item.label}</span>
                      {isActive && (
                        <span className="ml-auto w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)]" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Compact Footer Status */}
        <div className="pt-3 border-t border-white/[0.08] px-2">
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.06] flex items-center justify-between text-[11px] font-mono text-zinc-400 shadow-inner">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-zinc-300 font-medium">Cluster v2.4</span>
            </div>
            <span className="text-zinc-500">FastAPI</span>
          </div>
        </div>
      </aside>
    </>
  );
};
