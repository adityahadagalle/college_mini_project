import React, { useEffect, useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend,
} from 'recharts';
import { TrendingUp, PieChart as PieIcon, BarChart3, Calendar, RefreshCw, AlertCircle } from 'lucide-react';
import { apiService } from '../../services/api';
import { AnalyticsOverview, AnalyticsTrendsResponse, ChartDataset } from '../../types/sentiment';
import { SkeletonLoader } from '../common/SkeletonLoader';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';

const SENTIMENT_COLORS = {
  POSITIVE: '#10B981', // Emerald 500
  NEUTRAL: '#F59E0B',  // Amber 500
  NEGATIVE: '#F43F5E', // Rose 500
};

export const AnalyticsDashboard: React.FC = () => {
  const [period, setPeriod] = useState<'daily' | 'weekly' | 'monthly' | 'yearly'>('daily');
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null);
  const [trends, setTrends] = useState<AnalyticsTrendsResponse | null>(null);
  const [charts, setCharts] = useState<ChartDataset | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalyticsData = async (selectedPeriod: 'daily' | 'weekly' | 'monthly' | 'yearly') => {
    setLoading(true);
    setError(null);
    try {
      const [oData, tData, cData] = await Promise.all([
        apiService.getAnalyticsOverview(),
        apiService.getAnalyticsTrends(selectedPeriod),
        apiService.getAnalyticsCharts(),
      ]);

      setOverview(oData);
      setTrends(tData);
      setCharts(cData);
    } catch (err: any) {
      setError('Failed to fetch analytics from enterprise backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalyticsData(period);
  }, [period]);

  if (loading && !overview) {
    return (
      <div className="space-y-6 w-full">
        <SkeletonLoader variant="chart" count={2} />
      </div>
    );
  }

  // Format Pie chart data from backend
  const pieData = charts?.pie_chart
    ? charts.pie_chart.labels.map((label, idx) => ({
        name: label,
        value: charts.pie_chart.values[idx],
        color:
          label.toUpperCase() === 'POSITIVE'
            ? SENTIMENT_COLORS.POSITIVE
            : label.toUpperCase() === 'NEGATIVE'
            ? SENTIMENT_COLORS.NEGATIVE
            : SENTIMENT_COLORS.NEUTRAL,
      }))
    : [
        { name: 'Positive', value: overview?.positive_count || 0, color: SENTIMENT_COLORS.POSITIVE },
        { name: 'Neutral', value: overview?.neutral_count || 0, color: SENTIMENT_COLORS.NEUTRAL },
        { name: 'Negative', value: overview?.negative_count || 0, color: SENTIMENT_COLORS.NEGATIVE },
      ];

  // Format Bar chart data from backend
  const barData = charts?.bar_chart
    ? charts.bar_chart.labels.map((label, idx) => ({
        range: label,
        count: charts.bar_chart.values[idx],
      }))
    : [
        { range: '0.0 - 0.2', count: 2 },
        { range: '0.2 - 0.4', count: 5 },
        { range: '0.4 - 0.6', count: 12 },
        { range: '0.6 - 0.8', count: 28 },
        { range: '0.8 - 1.0', count: 53 },
      ];

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">Enterprise Analytics & Insights</h1>
            <Badge variant="zinc" size="sm">SQL Ingestion Pipeline</Badge>
          </div>
        </div>

        {/* Period Selector */}
        <div className="flex items-center gap-1.5 p-1 bg-[#07080C] border border-white/[0.08] shadow-inner rounded-lg text-xs">
          {(['daily', 'weekly', 'monthly', 'yearly'] as const).map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-3 py-1 rounded-md capitalize font-medium transition-all ${
                period === p
                  ? 'bg-gradient-to-b from-white via-zinc-100 to-zinc-200 text-zinc-950 font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.9)]'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.02]'
              }`}
            >
              {p}
            </button>
          ))}
          <button
            onClick={() => fetchAnalyticsData(period)}
            className="p-1 text-zinc-400 hover:text-zinc-100 rounded transition-colors ml-1"
            title="Refresh Analytics Data"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3.5 rounded-xl bg-rose-950/30 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Trend Line Chart */}
      <Card className="p-5 space-y-4">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
              <Calendar className="w-4 h-4 text-zinc-400" /> Sentiment Volume & Confidence Trends ({period.toUpperCase()})
            </h3>
            <p className="text-xs text-zinc-400 mt-0.5">
              Historical prediction volume over time categorized by sentiment label.
            </p>
          </div>
        </div>

        <div className="h-72 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trends?.trends || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="posGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={SENTIMENT_COLORS.POSITIVE} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={SENTIMENT_COLORS.POSITIVE} stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="neuGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={SENTIMENT_COLORS.NEUTRAL} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={SENTIMENT_COLORS.NEUTRAL} stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="negGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={SENTIMENT_COLORS.NEGATIVE} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={SENTIMENT_COLORS.NEGATIVE} stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" stroke="#71717A" fontSize={11} tickLine={false} />
              <YAxis stroke="#71717A" fontSize={11} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#090A0F',
                  borderColor: 'rgba(255,255,255,0.12)',
                  borderRadius: '10px',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)',
                  color: '#F4F4F5',
                  fontSize: '11px',
                }}
                itemStyle={{ color: '#F4F4F5' }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Area type="monotone" dataKey="positive" name="Positive" stroke={SENTIMENT_COLORS.POSITIVE} fillOpacity={1} fill="url(#posGrad)" />
              <Area type="monotone" dataKey="neutral" name="Neutral" stroke={SENTIMENT_COLORS.NEUTRAL} fillOpacity={1} fill="url(#neuGrad)" />
              <Area type="monotone" dataKey="negative" name="Negative" stroke={SENTIMENT_COLORS.NEGATIVE} fillOpacity={1} fill="url(#negGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Grid: Pie Chart & Confidence Bar Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sentiment Distribution Pie Chart */}
        <Card className="p-5 space-y-4">
          <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <PieIcon className="w-4 h-4 text-emerald-400" /> Sentiment Share Distribution
          </h3>
          <p className="text-xs text-zinc-400">Proportional ratio of positive, neutral, and negative predictions.</p>

          <div className="h-64 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={85}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#090A0F',
                    borderColor: 'rgba(255,255,255,0.12)',
                    borderRadius: '10px',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)',
                    color: '#F4F4F5',
                    fontSize: '11px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Confidence Score Bar Chart */}
        <Card className="p-5 space-y-4">
          <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-zinc-400" /> Confidence Score Distribution
          </h3>
          <p className="text-xs text-zinc-400">Frequency distribution of inference confidence probability bounds.</p>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="range" stroke="#71717A" fontSize={11} tickLine={false} />
                <YAxis stroke="#71717A" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#090A0F',
                    borderColor: 'rgba(255,255,255,0.12)',
                    borderRadius: '10px',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)',
                    color: '#F4F4F5',
                    fontSize: '11px',
                  }}
                />
                <Bar dataKey="count" name="Prediction Count" fill="#E4E4E7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
};
