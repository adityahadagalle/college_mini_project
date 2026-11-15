import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  LineChart,
  Line,
  Cell,
} from 'recharts';
import {
  Layers,
  Split,
  Heart,
  Boxes,
  TrendingDown,
  Filter,
  RefreshCw,
  AlertTriangle,
  X,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Info,
  Calendar,
  MessageSquare,
  ShieldCheck,
} from 'lucide-react';
import { apiService } from '../../services/api';
import {
  IntelligenceAnalyticsResponse,
  EvidenceDrilldownResponse,
  AspectSentimentByTopicItem,
  EmotionBySentimentItem,
  EmotionByAspectCategoryItem,
  TopicSummaryItem,
  NegativeAspectTrendItem,
} from '../../types/sentiment';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { SkeletonLoader } from '../common/SkeletonLoader';
import { EmptyState } from '../common/EmptyState';

const CX_COLORS: Record<string, string> = {
  'Frustration / Anger': '#F43F5E',
  'Disappointment': '#FB923C',
  'Delight / Excitement': '#10B981',
  'Satisfaction / Gratitude': '#38BDF8',
  'Hesitation / Confusion': '#F59E0B',
  'Neutral / Informative': '#A1A1AA',
};

export const CrossFeatureAnalytics: React.FC = () => {
  const [minSampleSize, setMinSampleSize] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(true);
  const [data, setData] = useState<IntelligenceAnalyticsResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Evidence Drilldown Drawer State
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const [drawerLoading, setDrawerLoading] = useState<boolean>(false);
  const [evidenceData, setEvidenceData] = useState<EvidenceDrilldownResponse | null>(null);
  const [filterTopicId, setFilterTopicId] = useState<string | undefined>(undefined);
  const [filterAspectTerm, setFilterAspectTerm] = useState<string | undefined>(undefined);
  const [filterEmotionCat, setFilterEmotionCat] = useState<string | undefined>(undefined);
  const [filterSentiment, setFilterSentiment] = useState<string | undefined>(undefined);
  const [evidencePage, setEvidencePage] = useState<number>(1);

  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await apiService.getCrossFeatureAnalytics(minSampleSize);
      setData(res);
    } catch (err: any) {
      setErrorMsg(
        err.response?.data?.detail || err.message || 'Failed to load cross-feature intelligence analytics.'
      );
    } finally {
      setLoading(false);
    }
  }, [minSampleSize]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  // Fetch drilldown evidence reviews
  const fetchEvidence = useCallback(
    async (page: number = 1) => {
      setDrawerLoading(true);
      try {
        const res = await apiService.getEvidenceDrilldown({
          topic_id: filterTopicId,
          aspect_term: filterAspectTerm,
          emotion_category: filterEmotionCat,
          sentiment: filterSentiment,
          page,
          page_size: 15,
        });
        setEvidenceData(res);
        setEvidencePage(page);
      } catch (err: any) {
        console.error('Failed to query evidence', err);
      } finally {
        setDrawerLoading(false);
      }
    },
    [filterTopicId, filterAspectTerm, filterEmotionCat, filterSentiment]
  );

  const openDrilldown = (params: {
    topicId?: string;
    aspectTerm?: string;
    emotionCategory?: string;
    sentiment?: string;
  }) => {
    setFilterTopicId(params.topicId);
    setFilterAspectTerm(params.aspectTerm);
    setFilterEmotionCat(params.emotionCategory);
    setFilterSentiment(params.sentiment);
    setEvidencePage(1);
    setDrawerOpen(true);
  };

  useEffect(() => {
    if (drawerOpen) {
      fetchEvidence(1);
    }
  }, [drawerOpen, fetchEvidence]);

  // Handle ESC key to close drawer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && drawerOpen) {
        setDrawerOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [drawerOpen]);

  // Chart data formatting for Aspect Sentiment by Topic
  const aspectTopicChartData = (data?.aspect_sentiment_by_topic || []).slice(0, 10).map((item) => ({
    label: `${item.topic_name.slice(0, 14)}: ${item.aspect_term}`,
    positive: item.positive_count,
    neutral: item.neutral_count,
    negative: item.negative_count,
    sample_count: item.sample_count,
    low_sample_warning: item.low_sample_warning,
    topic_id: item.topic_id,
    aspect_term: item.aspect_term,
  }));

  // Chart data formatting for Negative Trends
  const trendChartData = (data?.negative_aspect_trends || []).slice(0, 14).reverse().map((t) => ({
    date: t.date,
    aspect: t.aspect_term,
    negative_count: t.negative_count,
    rate_pct: Math.round(t.negative_rate * 100),
    sample_count: t.sample_count,
  }));

  return (
    <div className="space-y-6 w-full">
      {/* Sleek 56px Header and Control Bar */}
      <div className="bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl px-5 py-3.5 backdrop-blur-md flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner text-zinc-200">
            <Layers className="w-4 h-4 text-zinc-100" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-zinc-100 tracking-tight">
                Cross-Feature Intelligence
              </h1>
              <Badge variant="zinc" size="sm">
                SQL Multi-Model Relational Graph
              </Badge>
            </div>
            <p className="text-xs text-zinc-400 mt-0.5">
              Correlate sentiment polarities, aspect terms, CX emotion taxonomy, and discovered topic clusters across the entire review corpus.
            </p>
          </div>
        </div>

        {/* Control Toolbar */}
        <div className="flex items-center gap-2.5 self-start md:self-auto">
          <div className="flex items-center gap-1.5 bg-[#07080C] border border-white/[0.08] shadow-inner rounded-lg px-3 py-1.5 text-xs text-zinc-300">
            <Filter className="w-3.5 h-3.5 text-zinc-500" />
            <span className="text-[11px] text-zinc-400 font-mono">Min Sample (n):</span>
            <select
              value={minSampleSize}
              onChange={(e) => setMinSampleSize(Number(e.target.value))}
              className="bg-transparent text-xs text-zinc-200 font-semibold focus:outline-none cursor-pointer"
            >
              <option value={1} className="bg-zinc-900 text-zinc-200">1+ reviews</option>
              <option value={3} className="bg-zinc-900 text-zinc-200">3+ reviews</option>
              <option value={5} className="bg-zinc-900 text-zinc-200">5+ reviews</option>
              <option value={10} className="bg-zinc-900 text-zinc-200">10+ reviews</option>
            </select>
          </div>

          <Button
            onClick={fetchAnalytics}
            loading={loading}
            icon={RefreshCw}
            size="sm"
            variant="secondary"
          >
            Refresh
          </Button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-500/40 text-rose-300 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {loading ? (
        <SkeletonLoader variant="chart" count={3} />
      ) : data && data.total_analyzed_reviews > 0 ? (
        <div className="space-y-6">
          {/* KPI Summary Banner */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Card className="p-4 space-y-1">
              <span className="text-[10px] uppercase font-mono text-zinc-400 tracking-wider font-semibold">
                Corpus Population (N)
              </span>
              <p className="text-2xl font-bold font-mono tabular-nums text-zinc-100">
                {data.total_analyzed_reviews.toLocaleString()}
              </p>
              <span className="text-[10px] text-zinc-500">Total analyzed review records in database</span>
            </Card>

            <Card className="p-4 space-y-1">
              <span className="text-[10px] uppercase font-mono text-zinc-400 tracking-wider font-semibold">
                Aspect × Topic Co-Occurrences
              </span>
              <p className="text-2xl font-bold font-mono tabular-nums text-zinc-200">
                {data.aspect_sentiment_by_topic.length}
              </p>
              <span className="text-[10px] text-zinc-500">Active multidimensional relational pairs</span>
            </Card>

            <Card className="p-4 space-y-1">
              <span className="text-[10px] uppercase font-mono text-zinc-400 tracking-wider font-semibold">
                Discovered Topic Clusters
              </span>
              <p className="text-2xl font-bold font-mono tabular-nums text-emerald-400">
                {data.topic_sentiment_and_emotion_summary.length}
              </p>
              <span className="text-[10px] text-zinc-500">Unsupervised thematic clusters identified</span>
            </Card>
          </div>

          {/* Visualization 1: Aspect Sentiment by Topic */}
          <Card className="p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                  <Split className="w-4 h-4 text-zinc-400" />
                  Aspect Sentiment by Discovered Topic
                </h2>
                <p className="text-xs text-zinc-400">
                  Granular aspect polarities mapped within discovered thematic clusters. Click any row or bar to inspect supporting review evidence.
                </p>
              </div>
              <span className="text-[11px] font-mono tabular-nums text-zinc-500">
                Showing top {aspectTopicChartData.length} associations
              </span>
            </div>

            {/* Recharts Stacked Bar Chart with Obsidian Theme */}
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={aspectTopicChartData} margin={{ top: 10, right: 20, left: 0, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="label"
                    stroke="#71717A"
                    tick={{ fontSize: 10 }}
                    angle={-20}
                    textAnchor="end"
                  />
                  <YAxis stroke="#71717A" tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#090A0F',
                      borderColor: 'rgba(255,255,255,0.12)',
                      borderRadius: '10px',
                      boxShadow: '0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)',
                      color: '#F4F4F5',
                      fontSize: '11px',
                    }}
                    formatter={(value: any, name: any, item: any) => [
                      `${value} reviews (${Math.round((value / item.payload.sample_count) * 100)}%)`,
                      name === 'positive' ? 'Positive' : name === 'negative' ? 'Negative' : 'Neutral',
                    ]}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                  <Bar dataKey="positive" name="Positive" stackId="a" fill="#10B981" />
                  <Bar dataKey="neutral" name="Neutral" stackId="a" fill="#F59E0B" />
                  <Bar dataKey="negative" name="Negative" stackId="a" fill="#F43F5E" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Sortable Precision Data Table */}
            <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-[#07080C]/40 backdrop-blur-sm shadow-inner">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#090A0F]/90 text-zinc-400 uppercase font-mono text-[10px] border-b border-white/[0.08]">
                  <tr>
                    <th className="py-2.5 px-3">Discovered Topic</th>
                    <th className="py-2.5 px-3">Aspect Term</th>
                    <th className="py-2.5 px-3">Category</th>
                    <th className="py-2.5 px-3">Positive %</th>
                    <th className="py-2.5 px-3">Neutral %</th>
                    <th className="py-2.5 px-3">Negative %</th>
                    <th className="py-2.5 px-3">Sample Count</th>
                    <th className="py-2.5 px-3 text-right">Evidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {data.aspect_sentiment_by_topic.map((item, idx) => (
                    <tr
                      key={`${item.topic_id}-${item.aspect_term}-${idx}`}
                      onClick={() =>
                        openDrilldown({
                          topicId: item.topic_id,
                          aspectTerm: item.aspect_term,
                        })
                      }
                      className="hover:bg-white/[0.03] cursor-pointer transition-colors"
                    >
                      <td className="py-2 px-3 font-semibold text-zinc-200">
                        {item.topic_name}
                      </td>
                      <td className="py-2 px-3 font-mono text-zinc-300">
                        "{item.aspect_term}"
                      </td>
                      <td className="py-2 px-3 text-zinc-400">
                        {item.aspect_category || 'GENERAL'}
                      </td>
                      <td className="py-2 px-3 font-mono tabular-nums text-emerald-400">
                        {(item.positive_pct * 100).toFixed(0)}% ({item.positive_count})
                      </td>
                      <td className="py-2 px-3 font-mono tabular-nums text-amber-400">
                        {((item.neutral_count / Math.max(1, item.total_count)) * 100).toFixed(0)}% ({item.neutral_count})
                      </td>
                      <td className="py-2 px-3 font-mono tabular-nums text-rose-400">
                        {(item.negative_pct * 100).toFixed(0)}% ({item.negative_count})
                      </td>
                      <td className="py-2 px-3 font-mono tabular-nums text-zinc-300">
                        <div className="flex items-center gap-1.5">
                          <span>n = {item.sample_count}</span>
                          {item.low_sample_warning && (
                            <span className="px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 text-[9px] border border-amber-500/30 font-semibold">
                              Low Sample (n &lt; 5)
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-2 px-3 text-right">
                        <button className="text-zinc-400 hover:text-zinc-200 transition-colors">
                          <ExternalLink className="w-3.5 h-3.5 inline" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Visualizations Row 2: Emotion by Sentiment & Emotion by Aspect */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Emotion by Overall Sentiment */}
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                  <Heart className="w-4 h-4 text-rose-400" />
                  Emotion Distribution by Overall Sentiment
                </h3>
                <span className="text-[10px] font-mono text-zinc-500">Cross-Model CX Alignment</span>
              </div>

              <div className="space-y-3">
                {data.emotion_by_overall_sentiment.map((item, idx) => (
                  <div
                    key={idx}
                    onClick={() =>
                      openDrilldown({
                        sentiment: item.sentiment_label,
                        emotionCategory: item.emotion_category,
                      })
                    }
                    className="p-3.5 rounded-xl bg-[#090A0F]/80 border border-white/[0.06] hover:border-white/[0.14] hover:bg-white/[0.02] cursor-pointer transition-all space-y-2 shadow-xs"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            item.sentiment_label === 'POSITIVE'
                              ? 'emerald'
                              : item.sentiment_label === 'NEGATIVE'
                              ? 'rose'
                              : 'amber'
                          }
                          size="sm"
                        >
                          {item.sentiment_label}
                        </Badge>
                        <span className="text-zinc-300 font-semibold">{item.emotion_category}</span>
                      </div>
                      <span className="font-mono tabular-nums text-zinc-400 text-[11px]">
                        n = {item.sample_count} ({item.count} reviews)
                      </span>
                    </div>

                    <div className="w-full bg-[#07080C] border border-white/[0.06] rounded-full h-1.5 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(item.percentage * 100, 100)}%`,
                          backgroundColor: CX_COLORS[item.emotion_category] || '#38bdf8',
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] font-mono tabular-nums text-zinc-500">
                      <span>Rate: {(item.percentage * 100).toFixed(1)}%</span>
                      {item.low_sample_warning && (
                        <span className="text-amber-400 font-semibold">Low Sample (n &lt; 5)</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Emotion by Aspect Category */}
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                  <Split className="w-4 h-4 text-zinc-400" />
                  Emotion by Aspect Category
                </h3>
                <span className="text-[10px] font-mono text-zinc-500">Domain Driver Mapping</span>
              </div>

              <div className="space-y-3">
                {data.emotion_by_aspect_category.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-[#090A0F]/80 border border-white/[0.06] space-y-2 text-xs shadow-xs"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant="zinc" size="sm">
                          {item.aspect_category}
                        </Badge>
                        <span className="text-zinc-300 font-semibold">{item.emotion_category}</span>
                      </div>
                      <span className="font-mono tabular-nums text-zinc-400 text-[11px]">
                        n = {item.sample_count} ({item.count} reviews)
                      </span>
                    </div>

                    <div className="w-full bg-[#07080C] border border-white/[0.06] rounded-full h-1.5 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(item.percentage * 100, 100)}%`,
                          backgroundColor: CX_COLORS[item.emotion_category] || '#38bdf8',
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] font-mono tabular-nums text-zinc-500">
                      <span>Rate: {(item.percentage * 100).toFixed(1)}%</span>
                      {item.low_sample_warning && (
                        <span className="text-amber-400 font-semibold">Low Sample (n &lt; 5)</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* Visualization 3: Topic Sentiment & Emotion Summary */}
          <Card className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                  <Boxes className="w-4 h-4 text-zinc-400" />
                  Topic Sentiment &amp; Dominant Emotion Summary
                </h3>
                <p className="text-xs text-zinc-400">
                  Ranking discovered thematic clusters by volume, sentiment distribution, and primary emotional driver.
                </p>
              </div>
            </div>

            <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-[#07080C]/40 backdrop-blur-sm shadow-inner">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#090A0F]/90 text-zinc-400 uppercase font-mono text-[10px] border-b border-white/[0.08]">
                  <tr>
                    <th className="py-2.5 px-3">#</th>
                    <th className="py-2.5 px-3">Topic Title</th>
                    <th className="py-2.5 px-3">Review Volume</th>
                    <th className="py-2.5 px-3">Sentiment Breakdown (Pos / Neu / Neg)</th>
                    <th className="py-2.5 px-3">Dominant Emotion</th>
                    <th className="py-2.5 px-3 text-right">Inspect Evidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {data.topic_sentiment_and_emotion_summary.map((t) => (
                    <tr
                      key={t.topic_id}
                      onClick={() => openDrilldown({ topicId: t.topic_id })}
                      className="hover:bg-white/[0.03] cursor-pointer transition-colors"
                    >
                      <td className="py-2 px-3 font-mono text-zinc-500">{t.topic_index + 1}</td>
                      <td className="py-2 px-3 font-semibold text-zinc-200">{t.topic_name}</td>
                      <td className="py-2 px-3 font-mono tabular-nums text-zinc-200">
                        {t.total_reviews} reviews
                      </td>
                      <td className="py-2 px-3">
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-[#07080C] rounded-full h-1.5 overflow-hidden flex border border-white/[0.08]">
                            <div
                              style={{ width: `${(t.sentiment_breakdown.POSITIVE || 0) * 100}%` }}
                              className="bg-emerald-500 h-full"
                            />
                            <div
                              style={{ width: `${(t.sentiment_breakdown.NEUTRAL || 0) * 100}%` }}
                              className="bg-amber-500 h-full"
                            />
                            <div
                              style={{ width: `${(t.sentiment_breakdown.NEGATIVE || 0) * 100}%` }}
                              className="bg-rose-500 h-full"
                            />
                          </div>
                          <span className="font-mono tabular-nums text-[10px] text-zinc-400">
                            {Math.round((t.sentiment_breakdown.POSITIVE || 0) * 100)}% / {Math.round((t.sentiment_breakdown.NEUTRAL || 0) * 100)}% / {Math.round((t.sentiment_breakdown.NEGATIVE || 0) * 100)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className="px-2 py-0.5 rounded text-[11px] font-semibold font-mono"
                          style={{
                            backgroundColor: `${CX_COLORS[t.dominant_emotion] || '#38bdf8'}20`,
                            color: CX_COLORS[t.dominant_emotion] || '#38bdf8',
                          }}
                        >
                          {t.dominant_emotion}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right">
                        <button className="text-zinc-400 hover:text-zinc-200">
                          <ExternalLink className="w-3.5 h-3.5 inline" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Visualization 4: Negative Aspect Trends Over Time */}
          {trendChartData.length > 0 && (
            <Card className="p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-rose-400" />
                    Negative Aspect Trends Over Time
                  </h3>
                  <p className="text-xs text-zinc-400">
                    Daily frequency tracking of critical product friction points and negative review rates.
                  </p>
                </div>
              </div>

              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendChartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" stroke="#71717A" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#71717A" tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#090A0F',
                        borderColor: 'rgba(255,255,255,0.12)',
                        borderRadius: '10px',
                        boxShadow: '0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)',
                        color: '#F4F4F5',
                        fontSize: '11px',
                      }}
                      formatter={(val: any, name: any, item: any) => [
                        `${val} negative mentions (${item.payload.rate_pct}% of ${item.payload.sample_count} total)`,
                        `Aspect: ${item.payload.aspect}`,
                      ]}
                    />
                    <Line
                      type="monotone"
                      dataKey="negative_count"
                      stroke="#F43F5E"
                      strokeWidth={2}
                      dot={{ r: 3, fill: '#F43F5E' }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          {/* Methodological Disclaimer Footer */}
          <div className="p-3.5 rounded-xl border border-white/[0.08] bg-gradient-to-b from-[#141720]/80 to-[#0D0F14]/90 text-zinc-400 text-xs flex items-center gap-2.5 shadow-inner backdrop-blur-md">
            <ShieldCheck className="w-4 h-4 text-zinc-400 shrink-0" />
            <span className="text-[11px] leading-relaxed">
              <strong className="text-zinc-200">Statistical Methodology Disclosure:</strong> {data.methodology_note}
            </span>
          </div>
        </div>
      ) : (
        <EmptyState
          icon={<Layers className="w-6 h-6 text-zinc-400" />}
          title="No Cross-Feature Analytics Available"
          description="Analyze customer reviews in the Unified Review Intelligence tab or run batch ingestion to populate cross-model relational data."
        />
      )}

      {/* SLIDE-OVER EVIDENCE DRILLDOWN DRAWER */}
      {drawerOpen && (
        <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
          {/* Backdrop */}
          <div
            onClick={() => setDrawerOpen(false)}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm transition-opacity"
            aria-hidden="true"
          />

          {/* Drawer Panel */}
          <div className="relative w-full max-w-xl bg-gradient-to-b from-[#141720]/95 to-[#0A0C12]/98 border-l border-white/[0.08] shadow-[0_0_50px_rgba(0,0,0,0.8)] backdrop-blur-2xl flex flex-col h-full z-10 animate-scale-in">
            {/* Drawer Header */}
            <div className="p-5 border-b border-white/[0.08] flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-zinc-300" />
                  Review Evidence Drilldown
                </h3>
                <span className="text-xs text-zinc-400 font-mono tabular-nums">
                  {evidenceData?.total_matched || 0} matching customer reviews
                </span>
              </div>

              <button
                onClick={() => setDrawerOpen(false)}
                className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/[0.06] transition-colors"
                title="Close (Esc)"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Active Filter Chips */}
            <div className="p-3 bg-[#07080C]/80 border-b border-white/[0.06] flex flex-wrap items-center gap-2 text-xs">
              <span className="text-[10px] text-zinc-500 uppercase font-mono">Active Filters:</span>
              {filterTopicId && <Badge variant="zinc" size="sm">Topic Filtered</Badge>}
              {filterAspectTerm && <Badge variant="emerald" size="sm">Aspect: "{filterAspectTerm}"</Badge>}
              {filterEmotionCat && (
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-semibold font-mono"
                  style={{
                    backgroundColor: `${CX_COLORS[filterEmotionCat] || '#38bdf8'}20`,
                    color: CX_COLORS[filterEmotionCat] || '#38bdf8',
                  }}
                >
                  {filterEmotionCat}
                </span>
              )}
              {filterSentiment && (
                <Badge
                  variant={filterSentiment === 'POSITIVE' ? 'emerald' : filterSentiment === 'NEGATIVE' ? 'rose' : 'amber'}
                  size="sm"
                >
                  {filterSentiment}
                </Badge>
              )}
            </div>

            {/* Drawer Content */}
            <div className="flex-1 overflow-y-auto p-5 space-y-3">
              {drawerLoading ? (
                <SkeletonLoader variant="card" count={4} />
              ) : evidenceData && evidenceData.items.length > 0 ? (
                evidenceData.items.map((item, idx) => (
                  <div
                    key={`${item.prediction_id}-${idx}`}
                    className="p-4 rounded-xl border border-white/[0.06] bg-[#07080C]/60 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)] space-y-2 text-xs"
                  >
                    <p className="text-zinc-200 leading-relaxed font-sans italic">
                      "{item.review_text}"
                    </p>

                    <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-white/[0.06]">
                      {item.overall_sentiment && (
                        <Badge
                          variant={
                            item.overall_sentiment === 'POSITIVE'
                              ? 'emerald'
                              : item.overall_sentiment === 'NEGATIVE'
                              ? 'rose'
                              : 'amber'
                          }
                          size="sm"
                        >
                          {item.overall_sentiment}
                        </Badge>
                      )}

                      {item.aspect_term && (
                        <span className="px-2 py-0.5 rounded bg-zinc-850 border border-zinc-700 text-zinc-300 font-mono text-[10px]">
                          Aspect: {item.aspect_term}
                        </span>
                      )}

                      {item.emotion_category && (
                        <span
                          className="px-2 py-0.5 rounded text-[10px] font-semibold font-mono"
                          style={{
                            backgroundColor: `${CX_COLORS[item.emotion_category] || '#38bdf8'}20`,
                            color: CX_COLORS[item.emotion_category] || '#38bdf8',
                          }}
                        >
                          {item.emotion_category}
                        </span>
                      )}

                      {item.topic_name && (
                        <span className="text-zinc-400 text-[10px] truncate max-w-[150px]">
                          Topic: {item.topic_name}
                        </span>
                      )}
                    </div>

                    <div className="text-[10px] font-mono tabular-nums text-zinc-500 pt-1">
                      Prediction ID: {item.prediction_id.slice(0, 8)}... | Date: {item.created_at.slice(0, 10)}
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-xs text-zinc-500 italic text-center py-8">
                  No matching review evidence found for active filters.
                </p>
              )}
            </div>

            {/* Drawer Pagination Footer */}
            {evidenceData && evidenceData.total_matched > 15 && (
              <div className="p-4 border-t border-white/[0.08] bg-[#090A0F]/80 flex items-center justify-between text-xs">
                <span className="text-zinc-400 font-mono tabular-nums text-[11px]">
                  Page {evidencePage} of {Math.ceil(evidenceData.total_matched / 15)}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    disabled={evidencePage <= 1}
                    onClick={() => fetchEvidence(evidencePage - 1)}
                    className="p-1.5 rounded-lg border border-white/[0.08] bg-[#141720] disabled:opacity-40 text-zinc-300 hover:text-white hover:bg-white/[0.04] transition-colors"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <button
                    disabled={evidencePage >= Math.ceil(evidenceData.total_matched / 15)}
                    onClick={() => fetchEvidence(evidencePage + 1)}
                    className="p-1.5 rounded-lg border border-white/[0.08] bg-[#141720] disabled:opacity-40 text-zinc-300 hover:text-white hover:bg-white/[0.04] transition-colors"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
