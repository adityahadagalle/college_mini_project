import React, { useState, useEffect } from 'react';
import {
  Split,
  Search,
  Sparkles,
  BarChart3,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Layers,
  ArrowRight,
  Database,
  RefreshCw,
  Info,
} from 'lucide-react';
import { apiService } from '../../services/api';
import {
  ABSAResponse,
  AspectResult,
  AspectAnalyticsResponse,
  ABSAMetadataResponse,
} from '../../types/sentiment';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { SkeletonLoader } from '../common/SkeletonLoader';
import { EmptyState } from '../common/EmptyState';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
} from 'recharts';

const SAMPLE_REVIEWS = [
  'The camera is excellent, but the battery drains quickly.',
  'The screen is sharp and clear, though customer support was unhelpful.',
  'Decent food, but the ambiance was noisy and the service was slow.',
  'Build quality is solid, but the software has frequent crashes.',
];

export const AspectExplorer: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'analyzer' | 'analytics' | 'benchmark'>('analyzer');
  const [inputText, setInputText] = useState<string>(SAMPLE_REVIEWS[0]);
  const [saveToDb, setSaveToDb] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [analyzedResult, setAnalyzedResult] = useState<ABSAResponse | null>(null);
  const [selectedAspectIdx, setSelectedAspectIdx] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Analytics & Metadata states
  const [analyticsData, setAnalyticsData] = useState<AspectAnalyticsResponse | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState<boolean>(false);
  const [metadata, setMetadata] = useState<ABSAMetadataResponse | null>(null);
  const [metadataLoading, setMetadataLoading] = useState<boolean>(false);
  const [aspectSearchFilter, setAspectSearchFilter] = useState<string>('');
  const [sentimentFilter, setSentimentFilter] = useState<'ALL' | 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE'>('ALL');

  // Trigger analysis
  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await apiService.analyzeAspects(inputText, saveToDb);
      setAnalyzedResult(res);
      setSelectedAspectIdx(res.aspects.length > 0 ? 0 : null);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Failed to analyze review aspects.');
    } finally {
      setLoading(false);
    }
  };

  // Keyboard shortcut Ctrl+Enter
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (activeSubTab === 'analyzer') {
          handleAnalyze();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [inputText, saveToDb, activeSubTab]);

  // Fetch analytics data
  const fetchAnalytics = async () => {
    setAnalyticsLoading(true);
    try {
      const data = await apiService.getAspectAnalytics(50);
      setAnalyticsData(data);
    } catch (err: any) {
      console.error('Failed to load aspect analytics:', err);
    } finally {
      setAnalyticsLoading(false);
    }
  };

  // Fetch metadata
  const fetchMetadata = async () => {
    setMetadataLoading(true);
    try {
      const data = await apiService.getABSAMetadata();
      setMetadata(data);
    } catch (err: any) {
      console.error('Failed to load ABSA metadata:', err);
    } finally {
      setMetadataLoading(false);
    }
  };

  useEffect(() => {
    if (activeSubTab === 'analytics') {
      fetchAnalytics();
    } else if (activeSubTab === 'benchmark') {
      fetchMetadata();
    }
  }, [activeSubTab]);

  // Initial auto-analysis on mount for user convenience
  useEffect(() => {
    handleAnalyze();
  }, []);

  // Helper for rendering highlighted span elements in original review text
  const renderAnnotatedText = () => {
    if (!analyzedResult || !analyzedResult.text) return null;
    const text = analyzedResult.text;
    const aspects = analyzedResult.aspects;

    if (aspects.length === 0) {
      return <p className="text-sm text-zinc-300 leading-relaxed font-sans">{text}</p>;
    }

    const sorted = [...aspects]
      .filter((a) => a.start_char >= 0 && a.end_char <= text.length)
      .sort((a, b) => a.start_char - b.start_char);

    const elements: React.ReactNode[] = [];
    let lastIdx = 0;

    sorted.forEach((asp, i) => {
      if (asp.start_char > lastIdx) {
        elements.push(
          <span key={`txt-${lastIdx}-${asp.start_char}`} className="text-zinc-300">
            {text.slice(lastIdx, asp.start_char)}
          </span>
        );
      }

      const isSelected = selectedAspectIdx === i;
      const spanClass =
        asp.sentiment === 'POSITIVE'
          ? 'prodigy-span-positive'
          : asp.sentiment === 'NEGATIVE'
          ? 'prodigy-span-negative'
          : 'prodigy-span-neutral';

      elements.push(
        <button
          key={`asp-${i}-${asp.start_char}`}
          onClick={() => setSelectedAspectIdx(i)}
          className={`group inline-flex items-baseline mx-0.5 px-1 py-0.5 rounded cursor-pointer transition-all ${spanClass} ${
            isSelected
              ? 'ring-1 ring-zinc-200 bg-zinc-800/80'
              : 'hover:bg-zinc-800/40'
          }`}
          title={`Aspect: ${asp.aspect} | Polarity: ${asp.sentiment} (${(asp.confidence * 100).toFixed(1)}%)`}
        >
          <span className="font-medium text-zinc-100">{text.slice(asp.start_char, asp.end_char)}</span>
          <span className="ml-1 text-[9px] font-mono uppercase px-1 py-0.2 rounded bg-zinc-900/90 text-zinc-300 border border-zinc-700/60 leading-none">
            {asp.sentiment === 'POSITIVE' ? 'pos' : asp.sentiment === 'NEGATIVE' ? 'neg' : 'neu'}
          </span>
        </button>
      );
      lastIdx = Math.max(lastIdx, asp.end_char);
    });

    if (lastIdx < text.length) {
      elements.push(
        <span key={`txt-end-${lastIdx}`} className="text-zinc-300">
          {text.slice(lastIdx)}
        </span>
      );
    }

    return (
      <div className="text-sm leading-relaxed p-4 bg-[#07080C] rounded-xl border border-white/[0.08] shadow-[inset_0_2px_4px_rgba(0,0,0,0.4)] font-sans">
        {elements}
      </div>
    );
  };

  // Format chart data for Recharts
  const chartFrequencyData =
    analyticsData?.top_positive_aspects.slice(0, 8).map((pos) => {
      const negMatch = analyticsData.top_negative_aspects.find((n) => n.aspect.toLowerCase() === pos.aspect.toLowerCase());
      return {
        aspect: pos.aspect,
        positive: pos.positive_count,
        negative: negMatch ? negMatch.negative_count : 0,
        neutral: pos.neutral_count,
      };
    }) || [];

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Split className="w-4 h-4" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">Aspect-Based Sentiment (ABSA)</h1>
            <Badge variant="zinc" size="sm">DeBERTa-v3 ATE + ALSC</Badge>
          </div>
        </div>

        {/* Sub-tab Navigation */}
        <div className="flex items-center gap-1 bg-[#07080C] p-1 rounded-lg border border-white/[0.08] shadow-inner shrink-0">
          <button
            onClick={() => setActiveSubTab('analyzer')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeSubTab === 'analyzer'
                ? 'bg-gradient-to-b from-white via-zinc-100 to-zinc-200 text-zinc-950 font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.9)]'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.02]'
            }`}
          >
            Review Analyzer
          </button>
          <button
            onClick={() => setActiveSubTab('analytics')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeSubTab === 'analytics'
                ? 'bg-gradient-to-b from-white via-zinc-100 to-zinc-200 text-zinc-950 font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.9)]'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.02]'
            }`}
          >
            Aspect Analytics
          </button>
          <button
            onClick={() => setActiveSubTab('benchmark')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeSubTab === 'benchmark'
                ? 'bg-gradient-to-b from-white via-zinc-100 to-zinc-200 text-zinc-950 font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.9)]'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.02]'
            }`}
          >
            Model Benchmarks
          </button>
        </div>
      </div>

      {/* TAB 1: REVIEW ANALYZER */}
      {activeSubTab === 'analyzer' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Input and Span Visualizer */}
          <div className="lg:col-span-7 space-y-6">
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-zinc-400" />
                  <h3 className="text-sm font-semibold text-zinc-200">Customer Review Input</h3>
                </div>
                <div className="flex items-center gap-2 text-[11px] font-mono text-zinc-500 tabular-nums">
                  <span>{inputText.length} chars</span>
                  <span className="hidden sm:inline text-zinc-600">•</span>
                  <span className="hidden sm:inline text-zinc-500">Press Ctrl+Enter to run</span>
                </div>
              </div>

              {/* Sample Buttons */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-medium text-zinc-400">Quick Test Cases:</span>
                <div className="flex flex-wrap gap-1.5">
                  {SAMPLE_REVIEWS.map((sample, idx) => (
                    <button
                      key={idx}
                      onClick={() => setInputText(sample)}
                      className="text-left px-2.5 py-1 text-[11px] rounded-lg bg-[#141720]/80 hover:bg-white/[0.06] text-zinc-300 border border-white/[0.08] shadow-xs transition-colors"
                    >
                      "{sample.slice(0, 38)}..."
                    </button>
                  ))}
                </div>
              </div>

              {/* Textarea */}
              <textarea
                rows={4}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type or paste a multi-aspect review to evaluate (e.g. 'The camera is amazing, but the battery life is poor')..."
                className="w-full bg-[#07080C] border border-white/[0.08] focus:border-zinc-400 rounded-xl p-3.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 shadow-[inset_0_2px_4px_rgba(0,0,0,0.5)] transition-all font-sans leading-relaxed"
              />

              {/* Controls */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-zinc-800/80">
                <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={saveToDb}
                    onChange={(e) => setSaveToDb(e.target.checked)}
                    className="rounded border-zinc-700 bg-zinc-900 text-zinc-100 focus:ring-zinc-500"
                  />
                  <span>Persist to Review Aspects Database</span>
                </label>

                <Button
                  onClick={handleAnalyze}
                  loading={loading}
                  icon={Split}
                  size="md"
                  variant="primary"
                >
                  Analyze Aspects
                </Button>
              </div>

              {errorMsg && (
                <div className="p-3 bg-rose-950/30 border border-rose-800/60 rounded-lg flex items-center gap-2 text-xs text-rose-300">
                  <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                  <span>{errorMsg}</span>
                </div>
              )}
            </Card>

            {/* Interactive Span Visualizer */}
            {analyzedResult && (
              <Card className="p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-sm font-semibold text-zinc-200">Interactive Span Highlight</h3>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[11px] font-mono text-zinc-400 tabular-nums">
                      Latency: <span className="text-zinc-200 font-semibold">{analyzedResult.latency_ms.toFixed(1)} ms</span>
                    </span>
                    <Badge variant="zinc" size="sm" className="tabular-nums">
                      {analyzedResult.aspect_count} Aspect{analyzedResult.aspect_count !== 1 ? 's' : ''}
                    </Badge>
                  </div>
                </div>

                {renderAnnotatedText()}

                <div className="flex items-center justify-between text-[11px] text-zinc-400 px-1 pt-1">
                  <div className="flex items-center gap-3">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" /> Positive
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-amber-400" /> Neutral
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-rose-400" /> Negative
                    </span>
                  </div>
                  <span className="text-zinc-500 font-mono text-[10px]">Click any highlight to inspect details</span>
                </div>
              </Card>
            )}
          </div>

          {/* Right Column: Aspect Details & Decomposition */}
          <div className="lg:col-span-5 space-y-4">
            <div className="flex items-center justify-between px-1">
              <h3 className="text-sm font-semibold text-zinc-200">Decomposed Aspects</h3>
              {analyzedResult && (
                <span className="text-[11px] text-zinc-400">
                  Sentence Preview: <span className="font-semibold text-zinc-200 uppercase">{analyzedResult.overall_sentiment_preview}</span>
                </span>
              )}
            </div>

            {loading ? (
              <SkeletonLoader variant="table" count={3} />
            ) : analyzedResult && analyzedResult.aspects.length > 0 ? (
              <div className="space-y-3">
                {analyzedResult.aspects.map((asp: AspectResult, idx: number) => {
                  const isSelected = selectedAspectIdx === idx;
                  const polarityVariant =
                    asp.sentiment === 'POSITIVE'
                      ? 'positive'
                      : asp.sentiment === 'NEGATIVE'
                      ? 'negative'
                      : 'neutral';

                  return (
                    <div
                      key={idx}
                      onClick={() => setSelectedAspectIdx(idx)}
                      className={`p-4 rounded-xl border transition-all cursor-pointer ${
                        isSelected
                          ? 'bg-[#161920] border-zinc-500 shadow-sm'
                          : 'bg-[#0F1115] hover:bg-[#161920] border-zinc-800/90'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-sm text-zinc-100">{asp.aspect}</span>
                            {asp.aspect_category && (
                              <Badge variant="zinc" size="sm" title="Deterministic rule-based category normalization tag">
                                {asp.aspect_category} (Rule Tag)
                              </Badge>
                            )}
                          </div>
                          <p className="text-[11px] font-mono text-zinc-400 mt-0.5 tabular-nums">
                            Offsets: [{asp.start_char}:{asp.end_char}] • Evidence: "{asp.evidence_text}"
                          </p>
                        </div>

                        <div className="flex flex-col items-end">
                          <Badge variant={polarityVariant} size="sm">
                            {asp.sentiment}
                          </Badge>
                          <span className="text-[11px] font-mono text-zinc-300 mt-1 font-semibold tabular-nums">
                            {(asp.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {/* Probability Bars */}
                      <div className="space-y-1 pt-1">
                        <div className="flex items-center justify-between text-[10px] font-mono text-zinc-400 tabular-nums">
                          <span>Confidence Distribution</span>
                          <span>
                            +{(asp.probabilities.POSITIVE * 100).toFixed(0)}% • {(asp.probabilities.NEUTRAL * 100).toFixed(0)}% • -{(asp.probabilities.NEGATIVE * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-[#09090B] rounded-full flex overflow-hidden border border-zinc-800">
                          <div
                            style={{ width: `${asp.probabilities.POSITIVE * 100}%` }}
                            className="bg-emerald-500 transition-all duration-300"
                            title={`Positive: ${(asp.probabilities.POSITIVE * 100).toFixed(1)}%`}
                          />
                          <div
                            style={{ width: `${asp.probabilities.NEUTRAL * 100}%` }}
                            className="bg-amber-500 transition-all duration-300"
                            title={`Neutral: ${(asp.probabilities.NEUTRAL * 100).toFixed(1)}%`}
                          />
                          <div
                            style={{ width: `${asp.probabilities.NEGATIVE * 100}%` }}
                            className="bg-rose-500 transition-all duration-300"
                            title={`Negative: ${(asp.probabilities.NEGATIVE * 100).toFixed(1)}%`}
                          />
                        </div>
                      </div>

                      <div className="mt-3 pt-2 border-t border-zinc-800/80 flex items-center justify-between text-[10px] text-zinc-500 font-mono">
                        <span>Model: {asp.model_name} ({asp.model_version})</span>
                        <span>ATE BIO + ALSC Cross-Encoder</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : analyzedResult ? (
              <EmptyState
                icon={<Info className="w-5 h-5 text-zinc-400" />}
                title="No Explicit Aspects Found"
                description="The neural token classifier did not identify any explicit aspect targets in this sample."
              />
            ) : null}
          </div>
        </div>
      )}

      {/* TAB 2: ASPECT ANALYTICS */}
      {activeSubTab === 'analytics' && (
        <div className="space-y-6">
          {analyticsLoading ? (
            <SkeletonLoader variant="card" count={4} />
          ) : analyticsData ? (
            <>
              {/* Stat Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Total Recorded Aspects</span>
                  <p className="text-2xl font-bold font-mono text-zinc-100 tabular-nums">{analyticsData.total_aspects.toLocaleString()}</p>
                  <span className="text-[10px] text-zinc-500">Across saved review inferences</span>
                </Card>

                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Unique Aspect Targets</span>
                  <p className="text-2xl font-bold font-mono text-zinc-100 tabular-nums">{analyticsData.unique_aspects_count}</p>
                  <span className="text-[10px] text-zinc-500">Distinct entity & feature concepts</span>
                </Card>

                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider">Positive Aspect Ratio</span>
                  <p className="text-2xl font-bold font-mono text-emerald-400 tabular-nums">
                    {(analyticsData.sentiment_ratios.POSITIVE * 100).toFixed(1)}%
                  </p>
                  <span className="text-[10px] text-zinc-500 tabular-nums">{analyticsData.sentiment_distribution.POSITIVE} positive aspects</span>
                </Card>

                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-rose-400 uppercase tracking-wider">Negative Aspect Ratio</span>
                  <p className="text-2xl font-bold font-mono text-rose-400 tabular-nums">
                    {(analyticsData.sentiment_ratios.NEGATIVE * 100).toFixed(1)}%
                  </p>
                  <span className="text-[10px] text-zinc-500 tabular-nums">{analyticsData.sentiment_distribution.NEGATIVE} negative aspects</span>
                </Card>
              </div>

              {/* Chart & Comparison Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Recharts Bar Chart */}
                <div className="lg:col-span-7">
                  <Card className="p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-zinc-400" />
                        <h3 className="text-sm font-semibold text-zinc-200">Top Aspect Frequency & Polarity Distribution</h3>
                      </div>
                      <Button onClick={fetchAnalytics} variant="ghost" size="sm" icon={RefreshCw}>
                        Refresh
                      </Button>
                    </div>

                    <div className="h-72 w-full">
                      {chartFrequencyData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartFrequencyData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                            <XAxis dataKey="aspect" stroke="#71717A" fontSize={11} interval={0} angle={-25} textAnchor="end" />
                            <YAxis stroke="#71717A" fontSize={11} />
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
                            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                            <Bar dataKey="positive" name="Positive" fill="#10B981" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="neutral" name="Neutral" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="negative" name="Negative" fill="#EF4444" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-full flex items-center justify-center text-xs text-zinc-500">
                          No frequency data recorded yet. Run some sample analyses with 'Persist to Database'.
                        </div>
                      )}
                    </div>
                  </Card>
                </div>

                {/* Top Positive vs Top Negative Tables */}
                <div className="lg:col-span-5 space-y-6">
                  {/* Top Positive */}
                  <Card className="p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-emerald-400">Top Positive Aspects</h4>
                    </div>
                    <div className="divide-y divide-zinc-800 text-xs">
                      {analyticsData.top_positive_aspects.slice(0, 5).map((item, idx) => (
                        <div key={idx} className="py-2 flex items-center justify-between">
                          <span className="font-semibold text-zinc-200">{item.aspect}</span>
                          <div className="flex items-center gap-3 tabular-nums">
                            <span className="text-emerald-400 font-mono">+{item.positive_count}</span>
                            <span className="text-[10px] text-zinc-500 font-mono">Net: {item.net_sentiment_score}</span>
                          </div>
                        </div>
                      ))}
                      {analyticsData.top_positive_aspects.length === 0 && (
                        <p className="py-3 text-zinc-500 text-center text-xs">No positive aspects yet.</p>
                      )}
                    </div>
                  </Card>

                  {/* Top Negative */}
                  <Card className="p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-rose-400" />
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-rose-400">Top Critical / Negative Aspects</h4>
                    </div>
                    <div className="divide-y divide-zinc-800 text-xs">
                      {analyticsData.top_negative_aspects.slice(0, 5).map((item, idx) => (
                        <div key={idx} className="py-2 flex items-center justify-between">
                          <span className="font-semibold text-zinc-200">{item.aspect}</span>
                          <div className="flex items-center gap-3 tabular-nums">
                            <span className="text-rose-400 font-mono">-{item.negative_count}</span>
                            <span className="text-[10px] text-zinc-500 font-mono">Net: {item.net_sentiment_score}</span>
                          </div>
                        </div>
                      ))}
                      {analyticsData.top_negative_aspects.length === 0 && (
                        <p className="py-3 text-zinc-500 text-center text-xs">No negative aspects yet.</p>
                      )}
                    </div>
                  </Card>
                </div>
              </div>

              {/* Recent Records Table */}
              <Card className="p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-200">Recent Aspect Audit Trail</h3>
                    <p className="text-xs text-zinc-400">Recently extracted aspect terms stored in the database</p>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-zinc-500" />
                      <input
                        type="text"
                        placeholder="Search aspects..."
                        value={aspectSearchFilter}
                        onChange={(e) => setAspectSearchFilter(e.target.value)}
                        className="pl-8 pr-3 py-1.5 bg-[#09090B] border border-zinc-800 rounded-lg text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-zinc-500"
                      />
                    </div>

                    <select
                      value={sentimentFilter}
                      onChange={(e) => setSentimentFilter(e.target.value as any)}
                      className="bg-[#09090B] border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-zinc-500"
                    >
                      <option value="ALL">All Sentiments</option>
                      <option value="POSITIVE">Positive</option>
                      <option value="NEUTRAL">Neutral</option>
                      <option value="NEGATIVE">Negative</option>
                    </select>
                  </div>
                </div>

                <div className="overflow-x-auto rounded-xl border border-zinc-800">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#09090B] text-zinc-400 uppercase font-mono text-[10px] border-b border-zinc-800">
                      <tr>
                        <th className="py-3 px-4">Aspect Term</th>
                        <th className="py-3 px-4">Category</th>
                        <th className="py-3 px-4">Sentiment</th>
                        <th className="py-3 px-4">Confidence</th>
                        <th className="py-3 px-4">Span [start:end]</th>
                        <th className="py-3 px-4">Evidence Excerpt</th>
                        <th className="py-3 px-4">Recorded At</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/80 bg-[#0F1115]">
                      {analyticsData.recent_aspect_records
                        .filter((r) =>
                          (aspectSearchFilter === '' || r.aspect.toLowerCase().includes(aspectSearchFilter.toLowerCase())) &&
                          (sentimentFilter === 'ALL' || r.sentiment === sentimentFilter)
                        )
                        .map((row) => (
                          <tr key={row.id} className="hover:bg-zinc-800/30 transition-colors">
                            <td className="py-2.5 px-4 font-semibold text-zinc-200">{row.aspect}</td>
                            <td className="py-2.5 px-4 text-zinc-400 font-mono text-[11px]">{row.category ? `${row.category} (Rule Tag)` : 'N/A'}</td>
                            <td className="py-2.5 px-4">
                              <Badge
                                variant={
                                  row.sentiment === 'POSITIVE'
                                    ? 'positive'
                                    : row.sentiment === 'NEGATIVE'
                                    ? 'negative'
                                    : 'neutral'
                                }
                                size="sm"
                              >
                                {row.sentiment}
                              </Badge>
                            </td>
                            <td className="py-2.5 px-4 font-mono text-zinc-300 tabular-nums">{(row.confidence * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-4 font-mono text-zinc-400 text-[11px] tabular-nums">[{row.start_char}:{row.end_char}]</td>
                            <td className="py-2.5 px-4 text-zinc-300 italic max-w-xs truncate">"{row.evidence_text}"</td>
                            <td className="py-2.5 px-4 text-zinc-500 font-mono text-[11px] tabular-nums">
                              {row.created_at ? new Date(row.created_at).toLocaleTimeString() : 'N/A'}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          ) : (
            <EmptyState
              icon={<Database className="w-6 h-6 text-zinc-400" />}
              title="No Analytics Available"
              description="Analyze some customer reviews to generate aspect intelligence analytics."
            />
          )}
        </div>
      )}

      {/* TAB 3: BENCHMARK & ARCHITECTURE */}
      {activeSubTab === 'benchmark' && (
        <div className="space-y-6">
          {metadataLoading ? (
            <SkeletonLoader variant="card" count={3} />
          ) : metadata ? (
            <>
              {/* Benchmark Summary Header */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Benchmark Split</span>
                  <p className="text-base font-bold text-zinc-100">{metadata.benchmark_dataset}</p>
                  <span className="text-[10px] text-zinc-500">Leakage-deduplicated test partition (100 sentences, 266 aspects)</span>
                </Card>

                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider">ALSC Macro F1</span>
                  <p className="text-2xl font-bold font-mono text-emerald-300 tabular-nums">
                    {metadata.alsc_metrics.macro_f1.toFixed(4)}
                  </p>
                  <span className="text-[10px] text-emerald-400 font-medium tabular-nums">
                    vs Baseline: {metadata.baseline_comparison.baseline_macro_f1.toFixed(4)} (+124.9% gain)
                  </span>
                </Card>

                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">ATE Span F1</span>
                  <p className="text-2xl font-bold font-mono text-zinc-100 tabular-nums">
                    {metadata.ate_metrics.f1_score.toFixed(4)}
                  </p>
                  <span className="text-[10px] text-zinc-400 tabular-nums">
                    Precision: {metadata.ate_metrics.precision.toFixed(4)} • Recall: {metadata.ate_metrics.recall.toFixed(4)}
                  </span>
                </Card>
              </div>

              {/* Detailed Metrics Table */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="p-5 space-y-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-sm font-semibold text-zinc-200">ALSC Aspect-Sentiment Classification</h3>
                  </div>

                  <div className="space-y-3 text-xs">
                    <div className="flex justify-between py-1.5 border-b border-zinc-800">
                      <span className="text-zinc-400">Test Accuracy</span>
                      <span className="font-mono font-bold text-zinc-200 tabular-nums">
                        {(metadata.alsc_metrics.accuracy * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-zinc-800">
                      <span className="text-zinc-400">Baseline Accuracy</span>
                      <span className="font-mono text-zinc-400 tabular-nums">
                        {(metadata.baseline_comparison.baseline_accuracy * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-zinc-800">
                      <span className="text-zinc-400">Macro-Averaged F1</span>
                      <span className="font-mono font-bold text-emerald-400 tabular-nums">
                        {metadata.alsc_metrics.macro_f1.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-zinc-800">
                      <span className="text-zinc-400">Brier Calibration Score</span>
                      <span className="font-mono text-zinc-300 tabular-nums">
                        {metadata.alsc_metrics.brier_score.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </Card>

                <Card className="p-5 space-y-4">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-zinc-400" />
                    <h3 className="text-sm font-semibold text-zinc-200">Architecture & Hardware Configuration</h3>
                  </div>

                  <div className="space-y-3 text-xs">
                    <div className="flex justify-between py-1.5 border-b border-zinc-800">
                      <span className="text-zinc-400">Stage 1 Token Extractor (ATE)</span>
                      <span className="font-mono text-zinc-200">DeBERTa-v3 Token Classifier (BIO)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-zinc-800">
                      <span className="text-zinc-400">Stage 2 Polarity Classifier (ALSC)</span>
                      <span className="font-mono text-zinc-200">DeBERTa-v3 Cross-Encoder</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-zinc-800">
                      <span className="text-zinc-400">VRAM Partitioning</span>
                      <span className="font-mono text-zinc-300">PyTorch FP16 (4 GB RTX 3050 Safeguarded)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-zinc-800">
                      <span className="text-zinc-400">Legacy TensorFlow Isolation</span>
                      <span className="font-mono text-emerald-400">GPU Disabled (`CUDA_VISIBLE_DEVICES=""`)</span>
                    </div>
                  </div>
                </Card>
              </div>
            </>
          ) : (
            <EmptyState
              icon={<Cpu className="w-6 h-6 text-zinc-400" />}
              title="Benchmark Metrics Not Loaded"
              description="Unable to load model metadata from backend."
            />
          )}
        </div>
      )}
    </div>
  );
};
