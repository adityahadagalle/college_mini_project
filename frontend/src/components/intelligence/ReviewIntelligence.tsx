import React, { useState, useEffect, useRef } from 'react';
import {
  Brain,
  Zap,
  Split,
  Heart,
  Boxes,
  FileSpreadsheet,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ArrowRight,
  Database,
  RefreshCw,
  Info,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Search,
  Upload,
  Layers,
  ShieldAlert,
  CornerDownLeft,
} from 'lucide-react';
import { apiService } from '../../services/api';
import {
  UnifiedReviewResponse,
  UnifiedAspectItem,
  IntelligenceBatchJobStatus,
} from '../../types/sentiment';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import { SkeletonLoader } from '../common/SkeletonLoader';

interface ReviewIntelligenceProps {
  onNavigate?: (tab: string) => void;
}

const PRESET_REVIEWS = [
  {
    title: 'Battery & Screen Hardware',
    text: 'The battery life is fantastic and easily lasts two full days, but the screen glass scratches very easily with normal use.',
  },
  {
    title: 'Audio Quality & Customer Service',
    text: 'Noise cancellation and sound quality are truly top-notch, though customer support was completely unhelpful when I inquired about the warranty.',
  },
  {
    title: 'Delivery & Packaging',
    text: 'The package arrived three days early in pristine condition, and the setup was seamless.',
  },
  {
    title: 'Software Bugs & Overheating',
    text: 'Constant app crashes and the device overheats during charging. Extremely frustrating experience.',
  },
];

const CX_COLORS: Record<string, string> = {
  'Frustration / Anger': '#F43F5E',
  'Disappointment': '#FB923C',
  'Delight / Excitement': '#10B981',
  'Satisfaction / Gratitude': '#38BDF8',
  'Hesitation / Confusion': '#F59E0B',
  'Neutral / Informative': '#A1A1AA',
};

export const ReviewIntelligence: React.FC<ReviewIntelligenceProps> = ({ onNavigate }) => {
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');

  // Single review workbench state
  const [reviewText, setReviewText] = useState(PRESET_REVIEWS[0].text);
  const [saveToDb, setSaveToDb] = useState(true);
  const [projectTopic, setProjectTopic] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzedResult, setAnalyzedResult] = useState<UnifiedReviewResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [selectedAspect, setSelectedAspect] = useState<UnifiedAspectItem | null>(null);

  // Batch processor state
  const [batchInput, setBatchInput] = useState('');
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchJob, setBatchJob] = useState<IntelligenceBatchJobStatus | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const handleAnalyze = async () => {
    const text = reviewText.trim();
    if (text.length < 3) {
      setAnalysisError('Review text must be at least 3 characters long.');
      return;
    }
    setAnalyzing(true);
    setAnalysisError(null);
    setSelectedAspect(null);

    try {
      const res = await apiService.analyzeUnifiedReview(text, saveToDb, projectTopic);
      setAnalyzedResult(res);
    } catch (err: any) {
      setAnalysisError(
        err.response?.data?.detail || err.message || 'Unified multi-task intelligence request failed.'
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleAnalyze();
    }
  };

  const handleBatchSubmit = async () => {
    const lines = batchInput
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length >= 3);

    if (lines.length === 0) {
      setBatchError('Please provide at least one valid review with >= 3 characters.');
      return;
    }

    if (lines.length > 500) {
      setBatchError(`Batch size ${lines.length} exceeds maximum limit of 500 reviews.`);
      return;
    }

    setBatchLoading(true);
    setBatchError(null);
    if (pollingRef.current) clearInterval(pollingRef.current);

    try {
      const isSync = lines.length <= 25;
      const initialJob = await apiService.processUnifiedBatch(lines, saveToDb, isSync);
      setBatchJob(initialJob);

      if (!isSync && initialJob.status !== 'COMPLETED' && initialJob.status !== 'FAILED') {
        pollingRef.current = setInterval(async () => {
          try {
            const updated = await apiService.getUnifiedBatchStatus(initialJob.job_id);
            setBatchJob(updated);
            if (updated.status === 'COMPLETED' || updated.status === 'FAILED') {
              if (pollingRef.current) clearInterval(pollingRef.current);
            }
          } catch (pollErr) {
            console.error('Batch polling failed', pollErr);
          }
        }, 2000);
      }
    } catch (err: any) {
      setBatchError(err.response?.data?.detail || err.message || 'Batch intelligence processing failed.');
    } finally {
      setBatchLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      if (content) {
        setBatchInput(content);
      }
    };
    reader.readAsText(file);
  };

  // Helper to render Prodigy / Stripe style inline text highlights from aspect character offsets
  const renderInteractiveText = (text: string, aspects: UnifiedAspectItem[]) => {
    if (!aspects || aspects.length === 0) {
      return <p className="text-sm text-zinc-300 leading-relaxed font-sans">{text}</p>;
    }

    const sorted = [...aspects].sort((a, b) => a.start_char - b.start_char);
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;

    sorted.forEach((aspect, idx) => {
      const start = Math.max(lastIndex, Math.min(aspect.start_char, text.length));
      const end = Math.min(Math.max(start, aspect.end_char), text.length);

      if (start > lastIndex) {
        elements.push(
          <span key={`text-${lastIndex}`}>{text.slice(lastIndex, start)}</span>
        );
      }

      if (end > start) {
        const isSelected = selectedAspect === aspect;
        const styleClass =
          aspect.sentiment === 'POSITIVE'
            ? 'prodigy-span-positive'
            : aspect.sentiment === 'NEGATIVE'
            ? 'prodigy-span-negative'
            : 'prodigy-span-neutral';

        elements.push(
          <button
            key={`aspect-${idx}-${start}`}
            type="button"
            onClick={() => setSelectedAspect(isSelected ? null : aspect)}
            className={`inline-flex items-baseline px-1 mx-0.5 rounded-sm text-xs font-medium cursor-pointer transition-all duration-140 ${styleClass} ${
              isSelected ? 'ring-2 ring-zinc-300 ring-offset-1 ring-offset-[#09090B] font-semibold' : ''
            }`}
            title={`Aspect: ${aspect.aspect_term} (${aspect.sentiment} ${(aspect.confidence * 100).toFixed(0)}%)`}
          >
            <span>{text.slice(start, end)}</span>
            {aspect.aspect_category && (
              <span className="ml-1 text-[9px] font-mono uppercase tracking-wider opacity-80">
                {aspect.aspect_category}
              </span>
            )}
          </button>
        );
      }

      lastIndex = Math.max(lastIndex, end);
    });

    if (lastIndex < text.length) {
      elements.push(<span key={`text-tail`}>{text.slice(lastIndex)}</span>);
    }

    return <p className="text-sm text-zinc-200 leading-relaxed font-sans">{elements}</p>;
  };

  const getStatusIcon = (status: string) => {
    if (status === 'completed') return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />;
    if (status === 'unavailable') return <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />;
    if (status === 'pending') return <Clock className="w-3.5 h-3.5 text-zinc-400" />;
    return <XCircle className="w-3.5 h-3.5 text-rose-400" />;
  };

  const getStatusBadgeVariant = (status: string): 'emerald' | 'amber' | 'rose' | 'zinc' => {
    if (status === 'completed') return 'emerald';
    if (status === 'unavailable') return 'amber';
    if (status === 'pending') return 'zinc';
    return 'rose';
  };

  return (
    <div className="space-y-6 w-full">
      {/* Sleek 56px Page Header Bar */}
      <div className="bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 rounded-xl px-5 py-3.5 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] backdrop-blur-md flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-zinc-700 via-zinc-800 to-zinc-950 border border-white/[0.12] flex items-center justify-center text-zinc-100 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.15)]">
            <Brain className="w-4 h-4 text-zinc-100" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-tight">Unified Review Intelligence</h1>
              <Badge variant="zinc" size="sm">Multi-Model Pipeline</Badge>
            </div>
            <p className="text-xs text-zinc-400 mt-0.5">
              Simultaneous multi-task neural inference across Sentiment (BiLSTM), ABSA (DeBERTa-v3), Emotion (RoBERTa), and Topic Projection (MiniLM).
            </p>
          </div>
        </div>

        {/* Segmented Tab Switcher */}
        <div className="flex items-center gap-1 bg-[#07080C] p-1 rounded-xl border border-white/[0.06] shadow-inner self-start md:self-auto">
          <button
            onClick={() => setActiveTab('single')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'single'
                ? 'bg-gradient-to-b from-white to-zinc-200 text-zinc-950 font-semibold shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Single Review</span>
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'batch'
                ? 'bg-gradient-to-b from-white to-zinc-200 text-zinc-950 font-semibold shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]'
            }`}
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            <span>Batch Processor</span>
          </button>
        </div>
      </div>

      {/* TAB 1: SINGLE-REVIEW WORKBENCH */}
      {activeTab === 'single' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Input Workbench (Left Column: 5 Cols) */}
            <div className="lg:col-span-5 space-y-4">
              <Card className="p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider flex items-center gap-1.5 font-mono">
                    <Zap className="w-3.5 h-3.5 text-zinc-400" />
                    Review Input Workbench
                  </h2>
                  <span className="text-[11px] font-mono tabular-nums text-zinc-500">
                    {reviewText.length} / 5000 chars
                  </span>
                </div>

                <div className="relative">
                  <textarea
                    value={reviewText}
                    onChange={(e) => setReviewText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Paste or enter customer review text to execute full multi-task inference..."
                    className="w-full h-36 bg-[#07080C] border border-white/[0.08] focus:border-zinc-400 focus:ring-1 focus:ring-zinc-400/40 rounded-xl p-3.5 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none transition-all resize-none font-sans leading-relaxed shadow-inner"
                  />
                  <span className="absolute bottom-2.5 right-2.5 text-[10px] font-mono text-zinc-500 hidden sm:inline">
                    Ctrl + Enter to run
                  </span>
                </div>

                {/* Preset Domain Chips */}
                <div className="space-y-1.5">
                  <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 font-semibold">
                    Domain Test Presets
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {PRESET_REVIEWS.map((preset, idx) => (
                      <button
                        key={idx}
                        onClick={() => setReviewText(preset.text)}
                        className="px-2.5 py-1 rounded-lg bg-white/[0.03] hover:bg-white/[0.08] border border-white/[0.06] hover:border-white/[0.14] text-[11px] text-zinc-300 hover:text-white transition-all text-left shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)] active:scale-[0.98]"
                      >
                        {preset.title}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Execution Options */}
                <div className="pt-2 border-t border-white/[0.08] flex flex-wrap items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-4 text-zinc-400">
                    <label className="flex items-center gap-1.5 cursor-pointer text-xs">
                      <input
                        type="checkbox"
                        checked={saveToDb}
                        onChange={(e) => setSaveToDb(e.target.checked)}
                        className="rounded border-zinc-700 bg-zinc-900 text-zinc-100 focus:ring-0"
                      />
                      <span>Save to DB</span>
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer text-xs">
                      <input
                        type="checkbox"
                        checked={projectTopic}
                        onChange={(e) => setProjectTopic(e.target.checked)}
                        className="rounded border-zinc-700 bg-zinc-900 text-zinc-100 focus:ring-0"
                      />
                      <span>Project Topic</span>
                    </label>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      onClick={handleAnalyze}
                      loading={analyzing}
                      icon={Brain}
                      size="sm"
                      variant="primary"
                    >
                      Analyze Review
                    </Button>
                  </div>
                </div>

                {analysisError && (
                  <div className="p-3 rounded-lg bg-rose-950/30 border border-rose-500/40 text-rose-300 text-xs flex items-center gap-2">
                    <ShieldAlert className="w-4 h-4 shrink-0 text-rose-400" />
                    <span>{analysisError}</span>
                  </div>
                )}
              </Card>

              {/* Topic Prerequisite Callout Banner */}
              {analyzedResult && analyzedResult.capability_status.topics.status === 'unavailable' && (
                <div className="p-4 rounded-xl border border-amber-500/40 bg-amber-950/20 text-xs text-amber-200/90 space-y-2">
                  <div className="flex items-center gap-2 font-semibold text-amber-300">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                    <span>Topic Projection Prerequisite Required</span>
                  </div>
                  <p className="text-[11px] text-amber-300/80 leading-relaxed">
                    Live review topic projection requires at least one completed Topic Discovery run over &ge; 15 historical customer reviews to initialize centroid coordinates.
                  </p>
                  {onNavigate && (
                    <button
                      onClick={() => onNavigate('topics')}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-200 text-xs font-semibold transition-all mt-1"
                    >
                      <span>Explore &amp; Discover Topics</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Results Readout (Right Column: 7 Cols) */}
            <div className="lg:col-span-7 space-y-4">
              {analyzing ? (
                <SkeletonLoader variant="card" count={3} />
              ) : analyzedResult ? (
                <div className="space-y-4">
                  {/* Single Unified Capability Telemetry Strip (Shown once, not per card) */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                    {Object.entries(analyzedResult.capability_status).map(([capKey, cap]) => (
                      <div
                        key={capKey}
                        className="p-3 rounded-xl border border-white/[0.08] bg-gradient-to-b from-[#141720]/80 to-[#0D0F14]/90 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.06)] flex flex-col gap-1.5 backdrop-blur-sm"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] uppercase font-mono text-zinc-400 font-semibold tracking-wider">
                            {capKey}
                          </span>
                          {getStatusIcon(cap.status)}
                        </div>
                        <div className="flex items-baseline justify-between">
                          <span className="text-[11px] font-semibold text-zinc-200 capitalize">
                            {cap.model_name || cap.status}
                          </span>
                          {cap.latency_ms !== null && cap.latency_ms !== undefined && (
                            <span className="text-[10px] font-mono tabular-nums text-zinc-400">
                              {cap.latency_ms.toFixed(1)}ms
                            </span>
                          )}
                        </div>
                        {cap.error && (
                          <span className="text-[9px] text-amber-400 truncate" title={cap.error}>
                            {cap.error}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Interactive Evidence Highlighter Card */}
                  <Card className="p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-zinc-300" />
                        <h3 className="text-xs font-semibold text-zinc-200 uppercase tracking-wider font-mono">
                          Annotated Review Evidence
                        </h3>
                      </div>
                      <span className="text-[11px] font-mono tabular-nums text-zinc-400">
                        Total Latency: <span className="text-emerald-400 font-semibold">{analyzedResult.total_latency_ms.toFixed(1)} ms</span>
                      </span>
                    </div>

                    <div className="p-4 rounded-xl bg-[#07080C] border border-white/[0.08] shadow-[inset_0_2px_4px_rgba(0,0,0,0.5)] leading-relaxed">
                      {renderInteractiveText(analyzedResult.text_content, analyzedResult.aspects)}
                    </div>

                    {/* Selected Aspect Inspector Popover */}
                    {selectedAspect && (
                      <div className="p-3.5 rounded-xl bg-gradient-to-b from-[#141720]/95 to-[#0A0C12]/98 border border-white/[0.12] shadow-[0_8px_32px_rgba(0,0,0,0.6),inset_0_1px_0_rgba(255,255,255,0.08)] flex items-center justify-between text-xs animate-scale-in">
                        <div className="space-y-0.5">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-zinc-200">
                              Aspect: <span className="text-zinc-100 font-mono">"{selectedAspect.aspect_term}"</span>
                            </span>
                            {selectedAspect.aspect_category && (
                              <Badge variant="zinc" size="sm">
                                {selectedAspect.aspect_category}
                              </Badge>
                            )}
                          </div>
                          <span className="text-[10px] text-zinc-400 font-mono tabular-nums">
                            Offset: [{selectedAspect.start_char}..{selectedAspect.end_char}] | Sentiment: {selectedAspect.sentiment} ({(selectedAspect.confidence * 100).toFixed(1)}%)
                          </span>
                        </div>
                        <button
                          onClick={() => setSelectedAspect(null)}
                          className="text-zinc-400 hover:text-zinc-200 text-xs px-2.5 py-1 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner transition-colors"
                        >
                          Close
                        </button>
                      </div>
                    )}
                  </Card>

                  {/* Synchronized Readout Panels */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Panel 1: Overall Sentiment */}
                    <Card className="p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] uppercase font-semibold font-mono text-zinc-400 tracking-wider">
                          Overall Sentiment
                        </span>
                        {analyzedResult.overall_sentiment && (
                          <Badge
                            variant={
                              analyzedResult.overall_sentiment.sentiment === 'POSITIVE'
                                ? 'emerald'
                                : analyzedResult.overall_sentiment.sentiment === 'NEGATIVE'
                                ? 'rose'
                                : 'amber'
                            }
                            size="sm"
                          >
                            {analyzedResult.overall_sentiment.sentiment}
                          </Badge>
                        )}
                      </div>

                      {analyzedResult.overall_sentiment ? (
                        <div className="space-y-2">
                          <div className="flex items-baseline justify-between">
                            <span className="text-xl font-bold font-mono tabular-nums text-zinc-100">
                              {(analyzedResult.overall_sentiment.confidence * 100).toFixed(1)}%
                            </span>
                            <span className="text-[11px] font-mono text-zinc-400">
                              Confidence Score
                            </span>
                          </div>

                          {/* Tri-Color Probability Distribution Bar */}
                          <div className="w-full bg-[#07080C] rounded-full h-2 overflow-hidden flex border border-white/[0.08]">
                            <div
                              style={{
                                width: `${(analyzedResult.overall_sentiment.probabilities.positive || 0) * 100}%`,
                              }}
                              className="bg-emerald-500 h-full"
                              title={`Positive: ${((analyzedResult.overall_sentiment.probabilities.positive || 0) * 100).toFixed(1)}%`}
                            />
                            <div
                              style={{
                                width: `${(analyzedResult.overall_sentiment.probabilities.neutral || 0) * 100}%`,
                              }}
                              className="bg-amber-500 h-full"
                              title={`Neutral: ${((analyzedResult.overall_sentiment.probabilities.neutral || 0) * 100).toFixed(1)}%`}
                            />
                            <div
                              style={{
                                width: `${(analyzedResult.overall_sentiment.probabilities.negative || 0) * 100}%`,
                              }}
                              className="bg-rose-500 h-full"
                              title={`Negative: ${((analyzedResult.overall_sentiment.probabilities.negative || 0) * 100).toFixed(1)}%`}
                            />
                          </div>

                          <div className="flex justify-between text-[10px] font-mono tabular-nums text-zinc-500">
                            <span>Pos: {((analyzedResult.overall_sentiment.probabilities.positive || 0) * 100).toFixed(0)}%</span>
                            <span>Neu: {((analyzedResult.overall_sentiment.probabilities.neutral || 0) * 100).toFixed(0)}%</span>
                            <span>Neg: {((analyzedResult.overall_sentiment.probabilities.negative || 0) * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-xs text-zinc-500 italic">Sentiment service unavailable.</p>
                      )}
                    </Card>

                    {/* Panel 2: Customer Experience (CX) Emotion */}
                    <Card className="p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] uppercase font-semibold font-mono text-zinc-400 tracking-wider">
                          CX Emotion Rollup
                        </span>
                        {analyzedResult.emotions && (
                          <span
                            className="text-[11px] font-semibold px-2 py-0.5 rounded font-mono"
                            style={{
                              backgroundColor: `${CX_COLORS[analyzedResult.emotions.primary_cx_category] || '#38bdf8'}20`,
                              color: CX_COLORS[analyzedResult.emotions.primary_cx_category] || '#38bdf8',
                            }}
                          >
                            {analyzedResult.emotions.primary_cx_category}
                          </span>
                        )}
                      </div>

                      {analyzedResult.emotions ? (
                        <div className="space-y-2">
                          <div className="flex items-baseline justify-between">
                            <span className="text-xl font-bold font-mono tabular-nums text-zinc-100">
                              {(analyzedResult.emotions.confidence_score * 100).toFixed(1)}%
                            </span>
                            <span className="text-[11px] font-mono text-zinc-400">
                              CX Alignment Score
                            </span>
                          </div>

                          <div className="flex flex-wrap gap-1 pt-1">
                            {analyzedResult.emotions.detected_emotions.length > 0 ? (
                              analyzedResult.emotions.detected_emotions.map((emo, i) => (
                                <span
                                  key={i}
                                  className="px-2 py-0.5 rounded text-[10px] font-mono tabular-nums bg-[#141720] border border-white/[0.08] shadow-inner text-zinc-300"
                                >
                                  {emo.emotion} ({(emo.score * 100).toFixed(0)}%)
                                </span>
                              ))
                            ) : (
                              <span className="text-[10px] text-zinc-500">No active discrete emotion signals.</span>
                            )}
                          </div>
                        </div>
                      ) : (
                        <p className="text-xs text-zinc-500 italic">Emotion service unavailable.</p>
                      )}
                    </Card>

                    {/* Panel 3: Aspect Sentiments (ABSA) */}
                    <Card className="p-4 space-y-3 md:col-span-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Split className="w-3.5 h-3.5 text-zinc-400" />
                          <span className="text-[10px] uppercase font-semibold font-mono text-zinc-400 tracking-wider">
                            Extracted Aspects ({analyzedResult.aspects.length})
                          </span>
                        </div>
                        <span className="text-[11px] text-zinc-500 font-mono">
                          DeBERTa-v3 ABSA
                        </span>
                      </div>

                      {analyzedResult.aspects.length > 0 ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {analyzedResult.aspects.map((aspect, idx) => (
                            <div
                              key={idx}
                              onClick={() => setSelectedAspect(aspect)}
                              className={`p-2.5 rounded-lg border bg-zinc-950/60 cursor-pointer transition-all ${
                                selectedAspect === aspect
                                  ? 'border-zinc-500 ring-1 ring-zinc-500/50'
                                  : 'border-zinc-800 hover:border-zinc-700'
                              }`}
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-semibold text-xs text-zinc-200">
                                  {aspect.aspect_term}
                                </span>
                                <Badge
                                  variant={
                                    aspect.sentiment === 'POSITIVE'
                                      ? 'emerald'
                                      : aspect.sentiment === 'NEGATIVE'
                                      ? 'rose'
                                      : 'amber'
                                  }
                                  size="sm"
                                >
                                  {aspect.sentiment}
                                </Badge>
                              </div>
                              <div className="flex items-center justify-between text-[10px] text-zinc-400 font-mono tabular-nums">
                                <span>{aspect.aspect_category || 'GENERAL'}</span>
                                <span>{(aspect.confidence * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-xs text-zinc-500 italic py-2">
                          No distinct aspect terms extracted from this review.
                        </p>
                      )}
                    </Card>

                    {/* Panel 4: Projected Topic Cluster */}
                    <Card className="p-4 space-y-3 md:col-span-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Boxes className="w-3.5 h-3.5 text-zinc-400" />
                          <span className="text-[10px] uppercase font-semibold font-mono text-zinc-400 tracking-wider">
                            Projected Topic Cluster
                          </span>
                        </div>
                        {analyzedResult.topic && (
                          <Badge
                            variant={analyzedResult.topic.is_outlier ? 'amber' : 'zinc'}
                            size="sm"
                          >
                            {analyzedResult.topic.is_outlier ? 'Outlier (< 0.45 threshold)' : 'Cluster Assigned'}
                          </Badge>
                        )}
                      </div>

                      {analyzedResult.topic ? (
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-xl bg-[#07080C]/80 border border-white/[0.08] shadow-inner">
                          <div>
                            <span className="text-xs font-semibold text-zinc-200 block">
                              {analyzedResult.topic.topic_label}
                            </span>
                            <span className="text-[10px] text-zinc-400 font-mono">
                              Topic Index: {analyzedResult.topic.topic_index} | Discovery Version: {analyzedResult.topic.topic_run_version || 'Initial'}
                            </span>
                          </div>

                          <div className="text-right">
                            <span className="text-sm font-bold font-mono tabular-nums text-zinc-100 block">
                              {analyzedResult.topic.similarity_score !== null && analyzedResult.topic.similarity_score !== undefined
                                ? `${(analyzedResult.topic.similarity_score * 100).toFixed(1)}%`
                                : 'N/A'}
                            </span>
                            <span className="text-[10px] text-zinc-500 font-mono">
                              Centroid Cosine Match
                            </span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-xs text-zinc-500 italic py-1">
                          Topic projection inactive or no completed discovery runs found.
                        </p>
                      )}
                    </Card>
                  </div>
                </div>
              ) : (
                <EmptyState
                  icon={<Brain className="w-6 h-6 text-zinc-400" />}
                  title="Ready for Intelligence Analysis"
                  description="Click 'Analyze Review' or press Ctrl+Enter to orchestrate Sentiment, ABSA, Emotion, and Topic projection simultaneously."
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: BATCH PROCESSOR */}
      {activeTab === 'batch' && (
        <div className="space-y-6">
          <Card className="p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-zinc-200">
                  Unified Batch Intelligence Processor
                </h2>
                <p className="text-xs text-zinc-400">
                  Submit multiple customer reviews (up to 500) for memory-safe chunked multi-task inference.
                </p>
              </div>

              <label className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-xs font-medium text-zinc-200 cursor-pointer border border-zinc-800 hover:border-zinc-700 transition-all">
                <Upload className="w-3.5 h-3.5" />
                <span>Upload TXT / CSV</span>
                <input
                  type="file"
                  accept=".txt,.csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>

            <textarea
              value={batchInput}
              onChange={(e) => setBatchInput(e.target.value)}
              placeholder="Paste reviews here (one review per line)..."
              className="w-full h-44 bg-zinc-950/80 border border-zinc-800 rounded-lg p-3.5 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-zinc-500 focus:ring-1 focus:ring-zinc-500 transition-all font-mono leading-relaxed"
            />

            <div className="flex items-center justify-between pt-2">
              <span className="text-xs text-zinc-400 font-mono tabular-nums">
                {batchInput.split('\n').filter((l) => l.trim().length >= 3).length} valid reviews detected
              </span>

              <Button
                onClick={handleBatchSubmit}
                loading={batchLoading}
                icon={FileSpreadsheet}
                size="md"
                variant="primary"
              >
                Execute Batch Intelligence
              </Button>
            </div>

            {batchError && (
              <div className="p-3 rounded-lg bg-rose-950/30 border border-rose-500/40 text-rose-300 text-xs flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{batchError}</span>
              </div>
            )}
          </Card>

          {/* Active Batch Job Status Visualizer */}
          {batchJob && (
            <Card className="p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm text-zinc-200">
                    Job ID: <span className="font-mono text-zinc-100">{batchJob.job_id.slice(0, 8)}...</span>
                  </span>
                  <Badge variant={getStatusBadgeVariant(batchJob.status.toLowerCase())} size="sm">
                    {batchJob.status}
                  </Badge>
                </div>
                <span className="text-xs font-mono tabular-nums text-zinc-400">
                  Progress: {batchJob.progress_pct.toFixed(1)}%
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-zinc-950 rounded-full h-2 overflow-hidden border border-zinc-800">
                <div
                  className="bg-zinc-200 h-full transition-all duration-300 rounded-full"
                  style={{ width: `${Math.min(batchJob.progress_pct, 100)}%` }}
                />
              </div>

              {/* Metric Breakdown Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-center">
                  <span className="text-[10px] text-zinc-500 uppercase font-mono block">Total Records</span>
                  <span className="text-lg font-bold font-mono tabular-nums text-zinc-200">{batchJob.total_records}</span>
                </div>
                <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-500/25 text-center">
                  <span className="text-[10px] text-emerald-400 uppercase font-mono block">Succeeded</span>
                  <span className="text-lg font-bold font-mono tabular-nums text-emerald-300">{batchJob.succeeded_records}</span>
                </div>
                <div className="p-3 rounded-lg bg-amber-950/20 border border-amber-500/25 text-center">
                  <span className="text-[10px] text-amber-400 uppercase font-mono block">Partial</span>
                  <span className="text-lg font-bold font-mono tabular-nums text-amber-300">{batchJob.partial_records}</span>
                </div>
                <div className="p-3 rounded-lg bg-rose-950/20 border border-rose-500/25 text-center">
                  <span className="text-[10px] text-rose-400 uppercase font-mono block">Failed</span>
                  <span className="text-lg font-bold font-mono tabular-nums text-rose-300">{batchJob.failed_records}</span>
                </div>
              </div>

              {/* Synchronous Results Preview Table */}
              {batchJob.results && batchJob.results.length > 0 && (
                <div className="overflow-x-auto rounded-lg border border-zinc-800 mt-4">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-zinc-950 text-zinc-400 uppercase font-mono text-[10px] border-b border-zinc-800">
                      <tr>
                        <th className="py-2.5 px-3">#</th>
                        <th className="py-2.5 px-3">Review Text</th>
                        <th className="py-2.5 px-3">Sentiment</th>
                        <th className="py-2.5 px-3">Aspects</th>
                        <th className="py-2.5 px-3">CX Emotion</th>
                        <th className="py-2.5 px-3">Topic</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/80 bg-zinc-900/30">
                      {batchJob.results.map((r, i) => (
                        <tr key={r.review_id || i} className="hover:bg-zinc-800/40 transition-colors">
                          <td className="py-2 px-3 font-mono text-zinc-500">{i + 1}</td>
                          <td className="py-2 px-3 text-zinc-200 max-w-xs truncate" title={r.text_content}>
                            {r.text_content}
                          </td>
                          <td className="py-2 px-3">
                            {r.overall_sentiment ? (
                              <Badge
                                variant={
                                  r.overall_sentiment.sentiment === 'POSITIVE'
                                    ? 'emerald'
                                    : r.overall_sentiment.sentiment === 'NEGATIVE'
                                    ? 'rose'
                                    : 'amber'
                                }
                                size="sm"
                              >
                                {r.overall_sentiment.sentiment}
                              </Badge>
                            ) : (
                              <span className="text-zinc-500 font-mono text-[10px]">—</span>
                            )}
                          </td>
                          <td className="py-2 px-3 font-mono text-zinc-300">
                            {r.aspects.length > 0 ? (
                              <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-[10px]">
                                {r.aspects.map((a) => a.aspect_term).join(', ')}
                              </span>
                            ) : (
                              <span className="text-zinc-500 font-mono text-[10px]">None</span>
                            )}
                          </td>
                          <td className="py-2 px-3">
                            {r.emotions ? (
                              <span
                                className="px-1.5 py-0.5 rounded text-[10px] font-semibold font-mono"
                                style={{
                                  backgroundColor: `${CX_COLORS[r.emotions.primary_cx_category] || '#38bdf8'}20`,
                                  color: CX_COLORS[r.emotions.primary_cx_category] || '#38bdf8',
                                }}
                              >
                                {r.emotions.primary_cx_category}
                              </span>
                            ) : (
                              <span className="text-zinc-500 font-mono text-[10px]">—</span>
                            )}
                          </td>
                          <td className="py-2 px-3 text-[11px] text-zinc-300">
                            {r.topic ? (
                              <span className="truncate block max-w-[140px]" title={r.topic.topic_label}>
                                {r.topic.topic_label}
                              </span>
                            ) : (
                              <span className="text-zinc-500 font-mono text-[10px]">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}
        </div>
      )}
    </div>
  );
};
