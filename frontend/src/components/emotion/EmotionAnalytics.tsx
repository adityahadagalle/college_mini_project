import React, { useState, useEffect } from 'react';
import {
  Heart,
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
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { apiService } from '../../services/api';
import {
  EmotionResponse,
  EmotionAnalyticsResponse,
  EmotionMetadataResponse,
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
  CartesianGrid,
  Cell,
} from 'recharts';

const SAMPLE_REVIEWS = [
  {
    title: 'Concurrent Emotions (Defect + Fast Refund)',
    text: 'This device makes me furious and annoyed, but I am very grateful for the quick refund.',
  },
  {
    title: 'Delight / Excitement',
    text: 'I am absolutely thrilled and overjoyed with this amazing camera! Best purchase ever.',
  },
  {
    title: 'Negation Disappointment',
    text: 'I am not happy at all with the battery life, it drains within two hours.',
  },
  {
    title: 'Neutral / Informative',
    text: 'The device dimensions are 15 by 10 centimeters and it was shipped via standard post.',
  },
];

const CX_COLORS: Record<string, string> = {
  'Frustration / Anger': '#F43F5E', // rose-500
  'Disappointment': '#FB923C', // orange-400
  'Delight / Excitement': '#10B981', // emerald-500
  'Satisfaction / Gratitude': '#38BDF8', // sky-400
  'Hesitation / Confusion': '#F59E0B', // amber-500
  'Neutral / Informative': '#A1A1AA', // zinc-400
};

export const EmotionAnalytics: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'analyzer' | 'analytics' | 'benchmark'>('analyzer');
  const [inputText, setInputText] = useState(SAMPLE_REVIEWS[0].text);
  const [loading, setLoading] = useState(false);
  const [analyzedResult, setAnalyzedResult] = useState<EmotionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showFull28, setShowFull28] = useState(false);

  // Analytics tab state
  const [analyticsData, setAnalyticsData] = useState<EmotionAnalyticsResponse | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [cxFilter, setCxFilter] = useState('ALL');

  // Metadata tab state
  const [metadata, setMetadata] = useState<EmotionMetadataResponse | null>(null);
  const [metadataLoading, setMetadataLoading] = useState(false);

  useEffect(() => {
    if (activeSubTab === 'analytics') {
      fetchAnalytics();
    } else if (activeSubTab === 'benchmark') {
      fetchMetadata();
    }
  }, [activeSubTab]);

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await apiService.analyzeEmotion(inputText, true);
      setAnalyzedResult(res);
    } catch (err: any) {
      if (err.response?.status === 503) {
        setErrorMsg('Emotion Detection neural models are unavailable or not loaded (503 Service Unavailable).');
      } else {
        setErrorMsg(err.response?.data?.detail || err.message || 'Failed to analyze emotions.');
      }
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
  }, [inputText, activeSubTab]);

  const fetchAnalytics = async () => {
    setAnalyticsLoading(true);
    setErrorMsg(null);
    try {
      const data = await apiService.getEmotionAnalytics(100);
      setAnalyticsData(data);
    } catch (err: any) {
      if (err.response?.status === 503) {
        setErrorMsg('Emotion Detection neural models are unavailable (503).');
      } else {
        setErrorMsg(err.response?.data?.detail || 'Failed to fetch emotion analytics.');
      }
    } finally {
      setAnalyticsLoading(false);
    }
  };

  const fetchMetadata = async () => {
    setMetadataLoading(true);
    setErrorMsg(null);
    try {
      const meta = await apiService.getEmotionMetadata();
      setMetadata(meta);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || 'Failed to load emotion model metadata.');
    } finally {
      setMetadataLoading(false);
    }
  };

  const cxChartData = analyticsData
    ? Object.entries(analyticsData.cx_distribution).map(([category, count]) => ({
        category,
        count,
        fill: CX_COLORS[category] || '#71717A',
      }))
    : [];

  const rawEmotionsChartData = analyticsData?.top_raw_emotions.slice(0, 8).map((item) => ({
    label: item.label,
    count: item.count,
    category: item.cx_category,
  })) || [];

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Heart className="w-4 h-4 text-rose-400" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">Emotion Intelligence & CX Taxonomy</h1>
            <Badge variant="zinc" size="sm">28-Label GoEmotions + 6-CX Rollup</Badge>
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
            Emotion Analyzer
          </button>
          <button
            onClick={() => setActiveSubTab('analytics')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeSubTab === 'analytics'
                ? 'bg-gradient-to-b from-white via-zinc-100 to-zinc-200 text-zinc-950 font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.9)]'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.02]'
            }`}
          >
            CX Analytics
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

      {errorMsg && (
        <div className="p-3.5 bg-rose-950/30 border border-rose-800/60 rounded-xl flex items-center gap-3 text-xs text-rose-300">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <div className="flex-1">
            <span className="font-semibold">Service Notice: </span>
            <span>{errorMsg}</span>
          </div>
        </div>
      )}

      {/* TAB 1: EMOTION ANALYZER */}
      {activeSubTab === 'analyzer' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Input and Triggers */}
          <div className="lg:col-span-7 space-y-4">
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-200 uppercase tracking-wider">
                  Review Text Input
                </span>
                <span className="text-[11px] text-zinc-500 font-mono tabular-nums">
                  {inputText.length}/5000 chars • Ctrl+Enter
                </span>
              </div>

              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter customer feedback to evaluate emotion signals and CX rollup..."
                className="w-full h-32 bg-[#07080C] border border-white/[0.08] rounded-xl p-3.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-400 focus:ring-1 focus:ring-zinc-500 shadow-[inset_0_2px_4px_rgba(0,0,0,0.5)] transition-all resize-none leading-relaxed font-sans"
              />

              {/* Sample Review Pills */}
              <div className="space-y-1.5">
                <span className="text-[11px] text-zinc-400 font-medium">Quick Test Scenarios:</span>
                <div className="flex flex-wrap gap-1.5">
                  {SAMPLE_REVIEWS.map((sample, idx) => (
                    <button
                      key={idx}
                      onClick={() => setInputText(sample.text)}
                      className="px-2.5 py-1 rounded-lg bg-[#141720]/80 hover:bg-white/[0.06] border border-white/[0.08] text-[11px] text-zinc-300 hover:text-white transition-all text-left shadow-xs"
                    >
                      {sample.title}
                    </button>
                  ))}
                </div>
              </div>

              <div className="pt-2 flex justify-end border-t border-zinc-800/80">
                <Button
                  onClick={handleAnalyze}
                  loading={loading}
                  icon={Heart}
                  size="md"
                  variant="primary"
                >
                  Detect Emotions
                </Button>
              </div>
            </Card>

            {/* Active CX Rollup Breakdown */}
            {analyzedResult && (
              <Card className="p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-zinc-400" />
                    <h3 className="text-sm font-semibold text-zinc-200">Customer Experience (CX) Taxonomy Rollup</h3>
                  </div>
                  <span className="text-[11px] font-mono text-zinc-400 tabular-nums">
                    Latency: <span className="text-zinc-200 font-semibold">{analyzedResult.latency_ms.toFixed(1)} ms</span>
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {analyzedResult.detected_cx_categories.length > 0 ? (
                    analyzedResult.detected_cx_categories.map((cx, idx) => (
                      <div
                        key={idx}
                        className="p-3.5 rounded-lg border border-zinc-800/90 bg-[#161920] space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-xs text-zinc-200">{cx.category}</span>
                          <span
                            className="text-xs font-mono font-bold px-2 py-0.5 rounded tabular-nums"
                            style={{
                              backgroundColor: `${CX_COLORS[cx.category] || '#71717A'}20`,
                              color: CX_COLORS[cx.category] || '#71717A',
                            }}
                          >
                            {(cx.score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-[#09090B] rounded-full h-1.5 overflow-hidden border border-zinc-800">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.min(cx.score * 100, 100)}%`,
                              backgroundColor: CX_COLORS[cx.category] || '#71717A',
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-zinc-500 font-mono tabular-nums">
                          Calibrated Threshold: {(cx.threshold * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="col-span-2 p-3 bg-[#161920] rounded-lg border border-zinc-800 text-xs text-zinc-400 italic text-center">
                      No customer experience signals exceeded calibrated thresholds. Defaulting to Neutral / Informative.
                    </div>
                  )}
                </div>
              </Card>
            )}
          </div>

          {/* Right Column: Primary Emotion & Active Labels */}
          <div className="lg:col-span-5 space-y-4">
            <h3 className="text-sm font-semibold text-zinc-200 px-1">Primary Emotional Signal</h3>

            {loading ? (
              <SkeletonLoader variant="card" count={2} />
            ) : analyzedResult ? (
              <div className="space-y-4">
                {/* Primary Card */}
                <div
                  className="p-5 rounded-xl border transition-all space-y-3 bg-[#0F1115]"
                  style={{
                    borderColor: `${CX_COLORS[analyzedResult.primary_emotion] || '#3F3F46'}60`,
                  }}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
                      Primary CX Category
                    </span>
                    <Badge variant="zinc" size="sm">Multi-Label Sigmoid</Badge>
                  </div>

                  <p
                    className="text-xl font-bold tracking-tight"
                    style={{ color: CX_COLORS[analyzedResult.primary_emotion] || '#F4F4F5' }}
                  >
                    {analyzedResult.primary_emotion}
                  </p>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-zinc-400">Confidence Score:</span>
                      <span className="font-mono font-bold text-zinc-200 tabular-nums">
                        {(analyzedResult.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-[#09090B] rounded-full h-2 overflow-hidden border border-zinc-800">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.min(analyzedResult.confidence * 100, 100)}%`,
                          backgroundColor: CX_COLORS[analyzedResult.primary_emotion] || '#71717A',
                        }}
                      />
                    </div>
                  </div>

                  <div className="pt-2 border-t border-zinc-800 text-[11px] text-zinc-400 space-y-1 font-mono">
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Model:</span>
                      <span className="text-zinc-300">{analyzedResult.model_name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Version:</span>
                      <span className="text-zinc-300">{analyzedResult.model_version}</span>
                    </div>
                  </div>
                </div>

                {/* Active Discrete GoEmotions Labels */}
                <Card className="p-4 space-y-3">
                  <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                    Active GoEmotions Signals ({analyzedResult.detected_emotions.length})
                  </span>

                  {analyzedResult.detected_emotions.length > 0 ? (
                    <div className="space-y-2">
                      {analyzedResult.detected_emotions.map((item, idx) => (
                        <div
                          key={idx}
                          className="p-2.5 rounded-lg bg-[#161920] border border-zinc-800/90 flex items-center justify-between text-xs"
                        >
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-zinc-200">{item.label}</span>
                            <span className="text-[10px] text-zinc-400 font-mono">({item.cx_category})</span>
                          </div>
                          <div className="flex items-center gap-2 tabular-nums">
                            <span className="font-mono text-zinc-200 font-bold">
                              {(item.probability * 100).toFixed(1)}%
                            </span>
                            <span className="text-[10px] text-zinc-500 font-mono">
                              [th: {(item.threshold * 100).toFixed(0)}%]
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-zinc-400 italic">No discrete emotions exceeded per-class thresholds.</p>
                  )}

                  {/* 28-Label Full Drawer Toggle */}
                  <div className="pt-2 border-t border-zinc-800">
                    <button
                      onClick={() => setShowFull28(!showFull28)}
                      className="w-full flex items-center justify-between text-xs font-medium text-zinc-400 hover:text-zinc-200 py-1 transition-colors"
                    >
                      <span>Full 28-Label Probability Distribution</span>
                      {showFull28 ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>

                    {showFull28 && (
                      <div className="mt-3 space-y-1.5 max-h-60 overflow-y-auto pr-1">
                        {Object.entries(analyzedResult.raw_scores).map(([lbl, prob]) => (
                          <div key={lbl} className="flex items-center justify-between text-[11px] py-1 border-b border-zinc-800/40">
                            <span className="text-zinc-300 font-mono">{lbl}</span>
                            <div className="flex items-center gap-2 tabular-nums">
                              <div className="w-20 bg-[#09090B] rounded-full h-1.5 overflow-hidden border border-zinc-800">
                                <div
                                  className="h-full rounded-full bg-zinc-400"
                                  style={{ width: `${Math.min(prob * 100, 100)}%` }}
                                />
                              </div>
                              <span className="font-mono text-zinc-400 w-10 text-right">
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </Card>
              </div>
            ) : (
              <EmptyState
                icon={<Heart className="w-6 h-6 text-zinc-400" />}
                title="Awaiting Input"
                description="Click 'Detect Emotions' to run multi-label classification and see CX taxonomy rollup."
              />
            )}
          </div>
        </div>
      )}

      {/* TAB 2: ENTERPRISE ANALYTICS */}
      {activeSubTab === 'analytics' && (
        <div className="space-y-6">
          {analyticsLoading ? (
            <SkeletonLoader variant="table" count={4} />
          ) : analyticsData && analyticsData.total_records > 0 ? (
            <>
              {/* Summary Metric Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Total Recorded</span>
                  <p className="text-2xl font-bold font-mono text-zinc-100 tabular-nums">{analyticsData.total_records}</p>
                  <span className="text-[10px] text-zinc-500">Live reviews analyzed</span>
                </Card>

                <Card className="p-4 space-y-1 border-rose-900/40 bg-rose-950/10">
                  <span className="text-[11px] font-semibold text-rose-400 uppercase tracking-wider">Frustration / Anger</span>
                  <p className="text-2xl font-bold font-mono text-rose-300 tabular-nums">
                    {analyticsData.cx_distribution['Frustration / Anger'] || 0}
                  </p>
                  <span className="text-[10px] text-zinc-400 tabular-nums">
                    {((analyticsData.cx_ratios['Frustration / Anger'] || 0) * 100).toFixed(1)}% of total
                  </span>
                </Card>

                <Card className="p-4 space-y-1 border-emerald-900/40 bg-emerald-950/10">
                  <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider">Delight / Excitement</span>
                  <p className="text-2xl font-bold font-mono text-emerald-300 tabular-nums">
                    {analyticsData.cx_distribution['Delight / Excitement'] || 0}
                  </p>
                  <span className="text-[10px] text-zinc-400 tabular-nums">
                    {((analyticsData.cx_ratios['Delight / Excitement'] || 0) * 100).toFixed(1)}% of total
                  </span>
                </Card>

                <Card className="p-4 space-y-1 border-sky-900/40 bg-sky-950/10">
                  <span className="text-[11px] font-semibold text-sky-400 uppercase tracking-wider">Satisfaction / Gratitude</span>
                  <p className="text-2xl font-bold font-mono text-sky-300 tabular-nums">
                    {analyticsData.cx_distribution['Satisfaction / Gratitude'] || 0}
                  </p>
                  <span className="text-[10px] text-zinc-400 tabular-nums">
                    {((analyticsData.cx_ratios['Satisfaction / Gratitude'] || 0) * 100).toFixed(1)}% of total
                  </span>
                </Card>
              </div>

              {/* Charts Row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-zinc-200">CX Category Distribution</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={cxChartData} layout="vertical" margin={{ left: 80, right: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis type="number" stroke="#71717A" />
                        <YAxis dataKey="category" type="category" stroke="#A1A1AA" tick={{ fontSize: 11 }} />
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
                        <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                          {cxChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                <Card className="p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-zinc-200">Top Discrete GoEmotions Signals</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={rawEmotionsChartData} margin={{ bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="label" stroke="#A1A1AA" angle={-30} textAnchor="end" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#71717A" />
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
                        <Bar dataKey="count" fill="#F43F5E" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>

              {/* Records Table */}
              <Card className="p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <h3 className="text-sm font-semibold text-zinc-200">Recorded Customer Feedback</h3>

                  <div className="flex flex-wrap items-center gap-2">
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
                      <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Search records or emotions..."
                        className="bg-[#07080C] border border-white/[0.08] shadow-inner rounded-lg pl-8 pr-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-zinc-400 w-48 sm:w-56"
                      />
                    </div>
                    <select
                      value={cxFilter}
                      onChange={(e) => setCxFilter(e.target.value)}
                      className="bg-[#07080C] border border-white/[0.08] shadow-inner rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-zinc-400 cursor-pointer"
                    >
                      <option value="ALL">All CX Categories</option>
                      <option value="Frustration / Anger">Frustration / Anger</option>
                      <option value="Disappointment">Disappointment</option>
                      <option value="Delight / Excitement">Delight / Excitement</option>
                      <option value="Satisfaction / Gratitude">Satisfaction / Gratitude</option>
                      <option value="Hesitation / Confusion">Hesitation / Confusion</option>
                      <option value="Neutral / Informative">Neutral / Informative</option>
                    </select>
                  </div>
                </div>

                <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-[#07080C]/40 backdrop-blur-sm shadow-inner">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#090A0F]/90 text-zinc-400 uppercase font-mono text-[10px] border-b border-white/[0.08]">
                      <tr>
                        <th className="py-3 px-4">Record ID</th>
                        <th className="py-3 px-4">Primary CX Category</th>
                        <th className="py-3 px-4">Confidence</th>
                        <th className="py-3 px-4">Active Detected Emotions</th>
                        <th className="py-3 px-4">Logged At</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {analyticsData.recent_emotion_records
                        .filter((r) => {
                          const matchesCx = cxFilter === 'ALL' || r.primary_cx_category === cxFilter;
                          if (!matchesCx) return false;
                          if (!searchTerm.trim()) return true;
                          const term = searchTerm.toLowerCase();
                          const idMatch = r.id.toLowerCase().includes(term);
                          const catMatch = r.primary_cx_category.toLowerCase().includes(term);
                          const emoMatch = Array.isArray(r.detected_emotions) && r.detected_emotions.some(
                            (e: any) => (typeof e === 'string' ? e : e?.emotion || '').toLowerCase().includes(term)
                          );
                          return idMatch || catMatch || emoMatch;
                        })
                        .map((row) => (
                          <tr key={row.id} className="hover:bg-zinc-800/30 transition-colors">
                            <td className="py-2.5 px-4 font-mono text-zinc-400 text-[11px] tabular-nums">{row.id.slice(0, 8)}...</td>
                            <td className="py-2.5 px-4">
                              <span
                                className="px-2 py-0.5 rounded text-[11px] font-semibold"
                                style={{
                                  backgroundColor: `${CX_COLORS[row.primary_cx_category] || '#71717A'}20`,
                                  color: CX_COLORS[row.primary_cx_category] || '#71717A',
                                }}
                              >
                                {row.primary_cx_category}
                              </span>
                            </td>
                            <td className="py-2.5 px-4 font-mono text-zinc-300 tabular-nums">
                              {(row.confidence_score * 100).toFixed(1)}%
                            </td>
                            <td className="py-2.5 px-4">
                              <div className="flex flex-wrap gap-1">
                                {Array.isArray(row.detected_emotions) && row.detected_emotions.length > 0 ? (
                                  row.detected_emotions.map((e: any, i: number) => (
                                    <span key={i} className="px-1.5 py-0.5 bg-[#161920] border border-zinc-800 text-zinc-300 rounded text-[10px] font-mono">
                                      {e.label || e}
                                    </span>
                                  ))
                                ) : (
                                  <span className="text-zinc-500 text-[11px] italic">None</span>
                                )}
                              </div>
                            </td>
                            <td className="py-2.5 px-4 font-mono text-zinc-500 text-[11px] tabular-nums">
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
              title="No Emotion Analytics Available"
              description="Analyze customer reviews in the Emotion Analyzer tab to generate CX intelligence statistics."
            />
          )}
        </div>
      )}

      {/* TAB 3: BENCHMARK & CALIBRATION */}
      {activeSubTab === 'benchmark' && (
        <div className="space-y-6">
          {metadataLoading ? (
            <SkeletonLoader variant="card" count={3} />
          ) : metadata ? (
            <>
              {/* Benchmark Summary Header */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Benchmark Split</span>
                  <p className="text-sm font-bold text-zinc-100">{metadata.benchmark_dataset}</p>
                  <span className="text-[10px] text-zinc-500">Multi-label evaluated test partition</span>
                </Card>

                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider">Micro F1 Score</span>
                  <p className="text-2xl font-bold font-mono text-emerald-300 tabular-nums">
                    {metadata.metrics.micro_f1.toFixed(4)}
                  </p>
                  <span className="text-[10px] text-emerald-400 font-medium tabular-nums">
                    vs Prior: {metadata.metrics.baseline_comparison.micro_f1.toFixed(4)} (+{metadata.metrics.baseline_comparison.f1_improvement_pct.toFixed(1)}%)
                  </span>
                </Card>

                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Macro F1 Score</span>
                  <p className="text-2xl font-bold font-mono text-zinc-100 tabular-nums">
                    {metadata.metrics.macro_f1.toFixed(4)}
                  </p>
                  <span className="text-[10px] text-zinc-500 tabular-nums">
                    Hamming Loss: {metadata.metrics.hamming_loss.toFixed(4)}
                  </span>
                </Card>

                <Card className="p-4 space-y-1">
                  <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Brier Calibration</span>
                  <p className="text-2xl font-bold font-mono text-zinc-100 tabular-nums">
                    {metadata.metrics.mean_brier_score.toFixed(4)}
                  </p>
                  <span className="text-[10px] text-zinc-500 tabular-nums">
                    ECE: {metadata.metrics.expected_calibration_error.toFixed(4)}
                  </span>
                </Card>
              </div>

              {/* Domain Mismatch Note */}
              <div className="p-4 rounded-xl bg-[#0F1115] border border-zinc-800 space-y-2">
                <div className="flex items-center gap-2 text-zinc-300 text-xs font-semibold">
                  <Info className="w-4 h-4 text-zinc-400" />
                  <span>Domain Mismatch Analysis (Reddit Comments vs. Customer Reviews)</span>
                </div>
                <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                  <strong className="text-zinc-200">Source Domain:</strong> {metadata.domain_mismatch_analysis.source_domain}.<br />
                  <strong className="text-zinc-200">Target Domain:</strong> {metadata.domain_mismatch_analysis.target_domain}.<br />
                  <strong className="text-zinc-200">Mitigation Strategy:</strong> {metadata.domain_mismatch_analysis.mitigation_strategy}.
                </p>
              </div>

              {/* Per-Class Thresholds Table */}
              <Card className="p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-zinc-200">Per-Class Calibrated Thresholds & Test Metrics (28 GoEmotions Classes)</h3>
                  <Badge variant="zinc" size="sm">Tuned on Validation F1</Badge>
                </div>

                <div className="overflow-x-auto rounded-xl border border-zinc-800 max-h-96 overflow-y-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#09090B] text-zinc-400 uppercase font-mono text-[10px] sticky top-0 border-b border-zinc-800">
                      <tr>
                        <th className="py-2.5 px-3">Emotion Label</th>
                        <th className="py-2.5 px-3">Mapped CX Category</th>
                        <th className="py-2.5 px-3">Calibrated Threshold</th>
                        <th className="py-2.5 px-3">Precision</th>
                        <th className="py-2.5 px-3">Recall</th>
                        <th className="py-2.5 px-3">F1 Score</th>
                        <th className="py-2.5 px-3">Support</th>
                        <th className="py-2.5 px-3">Brier Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/80 bg-[#0F1115]">
                      {Object.entries(metadata.metrics.per_class).map(([label, metrics]) => (
                        <tr key={label} className="hover:bg-zinc-800/30 transition-colors font-mono text-[11px] tabular-nums">
                          <td className="py-2 px-3 font-semibold text-zinc-200">{label}</td>
                          <td className="py-2 px-3 text-zinc-400">{metadata.cx_taxonomy_mapping[label] || 'N/A'}</td>
                          <td className="py-2 px-3 text-zinc-200 font-bold">{metrics.threshold.toFixed(2)}</td>
                          <td className="py-2 px-3 text-zinc-300">{(metrics.precision * 100).toFixed(1)}%</td>
                          <td className="py-2 px-3 text-zinc-300">{(metrics.recall * 100).toFixed(1)}%</td>
                          <td className="py-2 px-3 text-emerald-400 font-bold">{(metrics.f1 * 100).toFixed(1)}%</td>
                          <td className="py-2 px-3 text-zinc-400">{metrics.support}</td>
                          <td className="py-2 px-3 text-zinc-300">{metrics.brier.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          ) : (
            <EmptyState
              icon={<Cpu className="w-6 h-6 text-zinc-400" />}
              title="Benchmarks Unavailable"
              description="Could not load emotion model calibration benchmarks."
            />
          )}
        </div>
      )}
    </div>
  );
};
