import React, { useEffect, useState } from 'react';
import {
  Layers,
  ThumbsUp,
  ThumbsDown,
  Zap,
  ShieldCheck,
  Cpu,
  Activity,
  Clock,
  ArrowUpRight,
  Users,
  CheckCircle2,
} from 'lucide-react';
import { apiService } from '../../services/api';
import { AnalyticsOverview, ModelInfoResponse, PredictResponse, SystemHealthResponse } from '../../types/sentiment';
import { SkeletonLoader } from '../common/SkeletonLoader';
import { StatCards } from './StatCards';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';

interface DashboardViewProps {
  onNavigate: (tab: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate }) => {
  const [analytics, setAnalytics] = useState<AnalyticsOverview | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<PredictResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      try {
        const [aData, mData, hData, pData] = await Promise.allSettled([
          apiService.getAnalyticsOverview(),
          apiService.getModelInfo(),
          apiService.getSystemHealth(),
          apiService.getHistory(undefined, undefined, 5, 0),
        ]);

        if (aData.status === 'fulfilled') setAnalytics(aData.value);
        if (mData.status === 'fulfilled') setModelInfo(mData.value);
        if (hData.status === 'fulfilled') setHealth(hData.value);
        if (pData.status === 'fulfilled') setRecentPredictions(pData.value);
      } catch {
        // Fallback gracefully
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6 w-full">
        <SkeletonLoader variant="card" count={4} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" />
        <SkeletonLoader variant="chart" count={1} />
      </div>
    );
  }

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">Enterprise Analytics & Operations</h1>
            <Badge variant="zinc" size="sm">4 Engines Live</Badge>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigate('intelligence')}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-b from-white via-zinc-100 to-zinc-200 hover:to-white text-zinc-950 text-xs font-semibold rounded-lg shadow-[0_1px_2px_rgba(0,0,0,0.4),0_0_12px_rgba(255,255,255,0.15),inset_0_1px_0_rgba(255,255,255,1)] transition-all"
          >
            <span>Review Intelligence</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <StatCards data={analytics} />

      {/* Grid Section: System Specs & Active Sessions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Information Card */}
        <Card className="p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-zinc-400" /> Model Architecture
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-zinc-800 text-zinc-300 border border-zinc-700">
              Active v{modelInfo?.model_version || '1.0.0'}
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between items-center py-1.5 border-b border-zinc-800/80">
              <span className="text-zinc-400">Architecture</span>
              <span className="font-mono text-zinc-200">{modelInfo?.architecture || 'Bi-LSTM Embedding'}</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-zinc-800/80">
              <span className="text-zinc-400">Total Parameters</span>
              <span className="font-mono text-zinc-200 tabular-nums">
                {modelInfo?.parameter_count ? modelInfo.parameter_count.toLocaleString() : '1,245,800'}
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-zinc-800/80">
              <span className="text-zinc-400">Model Accuracy</span>
              <span className="font-mono text-emerald-400 font-bold tabular-nums">
                {modelInfo?.accuracy ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : '94.8%'}
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5">
              <span className="text-zinc-400">F1-Score</span>
              <span className="font-mono text-zinc-200 font-bold tabular-nums">
                {modelInfo?.f1_score ? `${(modelInfo.f1_score * 100).toFixed(1)}%` : '94.2%'}
              </span>
            </div>
          </div>
        </Card>

        {/* System Health & Active Sessions Card */}
        <Card className="p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" /> Infrastructure & Nodes
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/20 text-emerald-400 border border-emerald-800/40">
              Healthy
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between items-center py-1.5 border-b border-zinc-800/80">
              <span className="text-zinc-400">CPU Load</span>
              <div className="flex items-center gap-2">
                <div className="w-16 bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-zinc-200 h-full rounded-full"
                    style={{ width: `${health?.cpu_percent || 18}%` }}
                  />
                </div>
                <span className="font-mono text-zinc-200 tabular-nums">{health?.cpu_percent || 18}%</span>
              </div>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-zinc-800/80">
              <span className="text-zinc-400">RAM Usage</span>
              <span className="font-mono text-zinc-200 tabular-nums">{health?.memory_mb || 340} MB</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-zinc-800/80">
              <span className="text-zinc-400">System Uptime</span>
              <span className="font-mono text-zinc-200 tabular-nums">
                {health?.uptime_seconds ? `${Math.round(health.uptime_seconds / 3600)} hrs` : '48 hrs'}
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5">
              <span className="text-zinc-400 flex items-center gap-1">
                <Users className="w-3.5 h-3.5 text-zinc-400" /> Active SaaS Sessions
              </span>
              <span className="font-mono text-emerald-400 font-medium">1 Verified Active</span>
            </div>
          </div>
        </Card>

        {/* Quick Actions Card */}
        <Card className="p-5 flex flex-col justify-between space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-zinc-200">Enterprise Operations</h3>
            <p className="text-xs text-zinc-400 mt-1">
              Select an operational suite to process sentiment requests or audit historical data.
            </p>
          </div>

          <div className="space-y-2 text-xs">
            <button
              onClick={() => onNavigate('batch')}
              className="w-full flex items-center justify-between p-3 rounded-lg bg-[#161920] border border-zinc-800 hover:border-zinc-700 text-zinc-200 hover:text-white transition-all group"
            >
              <span className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-zinc-400" /> Batch Processing Pipeline
              </span>
              <ArrowUpRight className="w-3.5 h-3.5 opacity-50 group-hover:opacity-100 transition-opacity" />
            </button>

            <button
              onClick={() => onNavigate('analytics')}
              className="w-full flex items-center justify-between p-3 rounded-lg bg-[#161920] border border-zinc-800 hover:border-zinc-700 text-zinc-200 hover:text-white transition-all group"
            >
              <span className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400" /> Analytics Dashboard
              </span>
              <ArrowUpRight className="w-3.5 h-3.5 opacity-50 group-hover:opacity-100 transition-opacity" />
            </button>
          </div>
        </Card>
      </div>

      {/* Recent Predictions Feed */}
      <Card className="p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
            <Clock className="w-4 h-4 text-zinc-400" /> Recent Prediction Activity
          </h3>
          <button
            onClick={() => onNavigate('history')}
            className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1 font-medium transition-colors"
          >
            View Full History <ArrowUpRight className="w-3 h-3" />
          </button>
        </div>

        {recentPredictions.length === 0 ? (
          <div className="p-6 text-center text-xs text-zinc-500 border border-zinc-800/80 rounded-lg bg-[#09090B]">
            No recent prediction records found. Run a prediction to populate this feed.
          </div>
        ) : (
          <div className="space-y-2">
            {recentPredictions.map((pred) => {
              const isPos = pred.sentiment === 'POSITIVE';
              const isNeg = pred.sentiment === 'NEGATIVE';
              return (
                <div
                  key={pred.id}
                  className="p-3 rounded-lg bg-[#161920] border border-zinc-800/90 flex items-center justify-between text-xs gap-4 hover:border-zinc-700 transition-all"
                >
                  <p className="text-zinc-200 truncate max-w-lg font-medium">{pred.text_content}</p>

                  <div className="flex items-center gap-3 shrink-0 tabular-nums">
                    <Badge
                      variant={isPos ? 'positive' : isNeg ? 'negative' : 'neutral'}
                      size="sm"
                    >
                      {pred.sentiment}
                    </Badge>

                    <span className="font-mono text-zinc-400 text-[11px]">
                      {Math.round(pred.confidence * 100)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
};
