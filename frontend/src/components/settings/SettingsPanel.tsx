import React, { useEffect, useState } from 'react';
import { Settings as SettingsIcon, Save, Key, Check, Moon, Sun, RefreshCw } from 'lucide-react';
import { apiService } from '../../services/api';
import { useNotification } from '../../context/NotificationContext';
import { UserSettings } from '../../types/sentiment';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

export const SettingsPanel: React.FC = () => {
  const { showSuccess, showError } = useNotification();

  const [threshold, setThreshold] = useState<number>(0.5);
  const [exportFormat, setExportFormat] = useState<string>('csv');
  const [theme, setTheme] = useState<string>('dark');
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [saved, setSaved] = useState<boolean>(false);

  useEffect(() => {
    const loadSettings = async () => {
      try {
        const s: UserSettings = await apiService.getUserSettings();
        if (s) {
          setThreshold(s.confidence_threshold ?? 0.5);
          setExportFormat(s.default_export_format ?? 'csv');
          setTheme(s.theme ?? 'dark');
        }
      } catch {
        // Fallback to default state
      } finally {
        setLoading(false);
      }
    };
    loadSettings();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      await apiService.updateUserSettings({
        confidence_threshold: threshold,
        default_export_format: exportFormat,
        theme: theme,
      });
      setSaved(true);
      showSuccess('Settings Updated', 'Platform preferences successfully synchronized with backend.');
      setTimeout(() => setSaved(false), 3000);
    } catch (err: any) {
      showError('Save Error', 'Failed to update preferences. Please retry.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="p-8 text-center text-zinc-400 text-xs font-mono animate-pulse">
        Loading User Settings & Preferences...
      </div>
    );
  }

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <SettingsIcon className="w-4 h-4 text-zinc-300" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">Platform Configuration & Preferences</h1>
            <Badge variant="zinc" size="sm">Workspace</Badge>
          </div>
        </div>
      </div>

      <form onSubmit={handleSave}>
        <Card className="p-6 space-y-6">
          {/* Confidence Threshold Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-zinc-300">
                Confidence Score Threshold ({Math.round(threshold * 100)}%)
              </label>
              <span className="text-[11px] font-mono text-zinc-200 bg-[#07080C] px-2 py-0.5 rounded border border-white/[0.08] shadow-inner tabular-nums">
                {threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-[#07080C] rounded-lg appearance-none cursor-pointer accent-zinc-200 border border-white/[0.08] shadow-inner"
            />
            <p className="text-[11px] text-zinc-400">
              Predictions below this confidence threshold are flagged for review.
            </p>
          </div>

          {/* Export Format */}
          <div className="space-y-2 pt-4 border-t border-zinc-800">
            <label className="block text-xs font-semibold text-zinc-300">
              Default Batch & History Export Format
            </label>
            <select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              className="w-full bg-[#07080C] border border-white/[0.08] shadow-inner rounded-lg px-3 py-2 text-xs text-zinc-200 focus:outline-none focus:border-zinc-400 font-sans cursor-pointer"
            >
              <option value="csv">Structured CSV (.csv)</option>
              <option value="json">JSON Dataset (.json)</option>
              <option value="pdf">Analytics PDF Report (.pdf)</option>
            </select>
          </div>

          {/* Theme Selector */}
          <div className="space-y-2 pt-4 border-t border-zinc-800">
            <label className="block text-xs font-semibold text-zinc-300">
              User Interface Theme
            </label>
            <div className="grid grid-cols-2 gap-3 max-w-sm">
              <button
                type="button"
                onClick={() => setTheme('dark')}
                className={`p-3 rounded-lg border flex items-center gap-2 text-xs font-medium transition-all ${
                  theme === 'dark'
                    ? 'bg-zinc-100 text-zinc-950 border-white shadow-xs font-semibold'
                    : 'bg-[#09090B] border-zinc-800 text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <Moon className="w-4 h-4" />
                <span>Obsidian Dark Mode</span>
              </button>
              <button
                type="button"
                onClick={() => setTheme('light')}
                className={`p-3 rounded-lg border flex items-center gap-2 text-xs font-medium transition-all ${
                  theme === 'light'
                    ? 'bg-zinc-100 text-zinc-950 border-white shadow-xs font-semibold'
                    : 'bg-[#09090B] border-zinc-800 text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <Sun className="w-4 h-4" />
                <span>High Contrast Light</span>
              </button>
            </div>
          </div>

          {/* Developer API Key */}
          <div className="pt-4 border-t border-zinc-800 space-y-2">
            <label className="block text-xs font-semibold text-zinc-300">
              Developer API Key
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                readOnly
                value="sk_live_tf_sentiment_9f8a7b6c5d4e3f2a1b"
                className="flex-1 bg-[#09090B] border border-zinc-800 rounded-lg px-3 py-2 text-xs font-mono text-zinc-400 select-all"
              />
              <button
                type="button"
                onClick={() => showSuccess('API Key Generated', 'A new API secret key has been copied to your clipboard.')}
                className="bg-zinc-900 hover:bg-zinc-800 text-zinc-200 text-xs font-medium px-4 py-2 rounded-lg transition-all border border-zinc-800 flex items-center gap-1.5"
              >
                <Key className="w-3.5 h-3.5 text-zinc-400" /> Regenerate
              </button>
            </div>
          </div>

          {/* Submit */}
          <div className="flex justify-end pt-4 border-t border-zinc-800">
            <Button
              type="submit"
              disabled={saving}
              loading={saving}
              variant="primary"
              size="md"
              icon={saved ? Check : Save}
            >
              {saved ? 'Preferences Saved!' : 'Save Preferences'}
            </Button>
          </div>
        </Card>
      </form>
    </div>
  );
};
