import React from 'react';
import { AnalyticsOverview } from '../../types/sentiment';
import { Layers, ThumbsUp, ThumbsDown, Zap } from 'lucide-react';
import { Card } from '../common/Card';

interface StatCardsProps {
  data: AnalyticsOverview | null;
}

export const StatCards: React.FC<StatCardsProps> = ({ data }) => {
  const cards = [
    {
      title: 'Total Inferences',
      value: (data?.total_predictions ?? 0).toLocaleString(),
      sub: 'Lifetime Sentiment Requests',
      icon: Layers,
      color: 'text-zinc-200',
      bg: 'bg-[#141720] border-white/[0.08] shadow-inner',
    },
    {
      title: 'Positive Ratio',
      value: `${((data?.sentiment_ratio.positive ?? 0) * 100).toFixed(1)}%`,
      sub: `${(data?.positive_count ?? 0).toLocaleString()} Positive Reviews`,
      icon: ThumbsUp,
      color: 'text-emerald-400',
      bg: 'bg-emerald-950/40 border-emerald-500/30 shadow-[inset_0_1px_0_0_rgba(16,185,129,0.2)]',
    },
    {
      title: 'Negative Ratio',
      value: `${((data?.sentiment_ratio.negative ?? 0) * 100).toFixed(1)}%`,
      sub: `${(data?.negative_count ?? 0).toLocaleString()} Negative Reviews`,
      icon: ThumbsDown,
      color: 'text-rose-400',
      bg: 'bg-rose-950/40 border-rose-500/30 shadow-[inset_0_1px_0_0_rgba(244,63,94,0.2)]',
    },
    {
      title: 'Average Latency',
      value: `${data?.average_latency_ms ?? 0} ms`,
      sub: 'Sub-300ms SLA Target',
      icon: Zap,
      color: 'text-zinc-200',
      bg: 'bg-[#141720] border-white/[0.08] shadow-inner',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <Card key={idx} hoverable className="space-y-3 p-4">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">{card.title}</span>
              <div className={`p-1.5 rounded-lg border ${card.bg}`}>
                <Icon className={`w-3.5 h-3.5 ${card.color}`} />
              </div>
            </div>
            <div>
              <div className="text-2xl font-bold text-zinc-100 font-mono tracking-tight tabular-nums">{card.value}</div>
              <div className="text-[11px] text-zinc-500 mt-0.5 tabular-nums">{card.sub}</div>
            </div>
          </Card>
        );
      })}
    </div>
  );
};
