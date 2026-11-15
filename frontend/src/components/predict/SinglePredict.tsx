import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Zap,
  Clock,
  AlertCircle,
  RefreshCw,
  Copy,
  Check,
  Download,
  FileText,
  Sparkles,
  X,
  Clipboard,
  Cpu,
  Activity,
  Calendar,
  Target,
  ChevronRight,
  ThumbsUp,
  ThumbsDown,
  Minus,
} from 'lucide-react';
import { apiService } from '../../services/api';
import { useNotification } from '../../context/NotificationContext';
import { PredictResponse } from '../../types/sentiment';
import { ProbabilityBar } from './ProbabilityBar';
import { ExplainabilityTags } from './ExplainabilityTags';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

const SAMPLE_REVIEWS = [
  {
    label: 'Positive',
    text: 'The customer service representative resolved my issue immediately. Outstanding support — 5 stars!',
    color: 'text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/5',
  },
  {
    label: 'Negative',
    text: 'This product stopped working after two days. Really disappointed with the poor quality and the slow shipping.',
    color: 'text-rose-400 border-rose-500/20 hover:bg-rose-500/5',
  },
  {
    label: 'Neutral',
    text: 'The package arrived on time. The item functions as described. Nothing exceptional to report.',
    color: 'text-amber-400 border-amber-500/20 hover:bg-amber-500/5',
  },
];

function getSentimentIcon(sentiment: string) {
  if (sentiment === 'POSITIVE') return <ThumbsUp className="w-5 h-5 text-emerald-400" />;
  if (sentiment === 'NEGATIVE') return <ThumbsDown className="w-5 h-5 text-rose-400" />;
  return <Minus className="w-5 h-5 text-amber-400" />;
}

function getConfidenceLabel(confidence: number): { label: string; color: string } {
  if (confidence >= 0.9) return { label: 'Very High', color: 'text-emerald-400' };
  if (confidence >= 0.75) return { label: 'High', color: 'text-emerald-400' };
  if (confidence >= 0.55) return { label: 'Moderate', color: 'text-amber-400' };
  return { label: 'Low', color: 'text-rose-400' };
}

export const SinglePredict: React.FC = () => {
  const { showSuccess, showError } = useNotification();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const [text, setText] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 320)}px`;
    }
  }, [text]);

  const handlePredict = useCallback(
    async (e?: React.FormEvent) => {
      if (e) e.preventDefault();
      const trimmed = text.trim();
      if (!trimmed) {
        setError('Please enter text to analyze.');
        return;
      }
      if (trimmed.length > 5000) {
        setError('Text exceeds maximum length of 5000 characters.');
        return;
      }

      setLoading(true);
      setError(null);
      setResult(null);
      try {
        const res = await apiService.predictSentiment(trimmed);
        setResult(res);
        showSuccess(
          'Analysis Complete',
          `Classified as ${res.sentiment} (${Math.round(res.confidence * 100)}% confidence)`,
        );
      } catch (err: any) {
        const msg =
          err.response?.data?.detail || 'Failed to analyze sentiment. Please try again.';
        setError(msg);
        showError('Prediction Error', msg);
      } finally {
        setLoading(false);
      }
    },
    [text, showSuccess, showError],
  );

  // Ctrl+Enter keyboard shortcut
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        handlePredict();
      }
    },
    [handlePredict],
  );

  const handlePaste = useCallback(async () => {
    try {
      const clipText = await navigator.clipboard.readText();
      setText(clipText);
      if (error) setError(null);
    } catch {
      showError('Paste Failed', 'Could not read clipboard. Please paste manually.');
    }
  }, [error, showError]);

  const handleCopyResult = useCallback(() => {
    if (!result) return;
    const summary = [
      `Sentiment: ${result.sentiment}`,
      `Confidence: ${(result.confidence * 100).toFixed(1)}%`,
      `Positive: ${(result.probabilities.positive * 100).toFixed(1)}%`,
      `Neutral:  ${(result.probabilities.neutral * 100).toFixed(1)}%`,
      `Negative: ${(result.probabilities.negative * 100).toFixed(1)}%`,
      `Latency:  ${result.latency_ms} ms`,
      `Text: "${result.text_content}"`,
    ].join('\n');
    navigator.clipboard.writeText(summary);
    setCopied(true);
    showSuccess('Copied', 'Prediction summary copied to clipboard.');
    setTimeout(() => setCopied(false), 2000);
  }, [result, showSuccess]);

  const handleExportJson = useCallback(() => {
    if (!result) return;
    const exportData = {
      ...result,
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sentiment_${result.id || 'prediction'}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showSuccess('Exported', 'Prediction exported as JSON file.');
  }, [result, showSuccess]);

  return (
    <div className="space-y-6 w-full">
      {/* Sleek 56px Header Bar */}
      <div className="bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl px-5 py-3.5 backdrop-blur-md flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner text-zinc-200">
            <Zap className="w-4 h-4 text-zinc-100" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-zinc-100 tracking-tight">Single Text Sentiment Inference</h1>
              <Badge variant="zinc" size="sm">&lt;300ms SLA</Badge>
            </div>
            <p className="text-xs text-zinc-400 mt-0.5">
              Real-time TensorFlow Bi-LSTM classification with token-level gradient attribution heatmaps.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-zinc-400">
          <kbd className="px-1.5 py-0.5 bg-[#07080C] border border-white/[0.08] shadow-inner rounded text-[11px] text-zinc-300">
            Ctrl+Enter
          </kbd>
          <span>to analyze</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <form onSubmit={handlePredict} className="space-y-3">
            <div className="rounded-xl border border-white/[0.08] bg-[#07080C] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.5),inset_0_1px_0_0_rgba(255,255,255,0.04)] overflow-hidden">
              {/* Input Toolbar */}
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/[0.08] bg-[#090A0F]/90">
                <label className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider font-mono">
                  Input Review
                </label>
                <div className="flex items-center gap-3 text-[11px] font-mono tabular-nums">
                  <span className="text-zinc-500">{wordCount} words</span>
                  <span
                    className={
                      text.length > 4500
                        ? 'text-rose-400'
                        : text.length > 4000
                        ? 'text-amber-400'
                        : 'text-zinc-500'
                    }
                  >
                    {text.length} / 5000
                  </span>
                </div>
              </div>

              <textarea
                ref={textareaRef}
                value={text}
                onChange={(e) => {
                  setText(e.target.value);
                  if (error) setError(null);
                }}
                onKeyDown={handleKeyDown}
                placeholder="Type or paste customer review, product feedback, support ticket…"
                rows={5}
                className="w-full bg-transparent px-4 py-3.5 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none resize-none min-h-[130px] leading-relaxed font-sans"
                style={{ overflow: 'hidden' }}
              />

              {/* Action bar */}
              <div className="flex items-center justify-between px-4 py-2.5 border-t border-white/[0.08] bg-[#090A0F]/60">
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={handlePaste}
                    className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 px-2.5 py-1 rounded-md hover:bg-zinc-800 transition-colors font-mono"
                  >
                    <Clipboard className="w-3.5 h-3.5" /> Paste
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setText('');
                      setResult(null);
                      if (error) setError(null);
                    }}
                    disabled={!text}
                    className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-rose-400 disabled:opacity-30 px-2.5 py-1 rounded-md hover:bg-zinc-800 transition-colors font-mono"
                  >
                    <X className="w-3.5 h-3.5" /> Clear
                  </button>
                </div>

                <Button
                  type="submit"
                  disabled={loading || !text.trim()}
                  loading={loading}
                  icon={Sparkles}
                  size="sm"
                  variant="primary"
                >
                  Analyze Sentiment
                </Button>
              </div>
            </div>
          </form>

          {/* Error Alert */}
          {error && (
            <div
              role="alert"
              className="flex items-center gap-2.5 p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs"
            >
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          {/* Sample Review Chips */}
          <div className="rounded-xl border border-zinc-800 bg-[#0F1115] p-4 space-y-2.5">
            <p className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wider font-mono">
              Try a sample review
            </p>
            <div className="flex flex-col gap-1.5">
              {SAMPLE_REVIEWS.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setText(sample.text);
                    if (error) setError(null);
                    setResult(null);
                  }}
                  className={`text-left text-xs px-3 py-2 rounded-lg transition-all border bg-zinc-950/60 hover:bg-zinc-900 ${sample.color}`}
                >
                  <span className="font-semibold mr-2 font-mono uppercase text-[10px]">{sample.label}</span>
                  <span className="text-zinc-300 truncate">{sample.text}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Results Panel (5 cols) */}
        <div className="lg:col-span-5">
          {loading ? (
            <div className="rounded-xl border border-zinc-800 bg-[#0F1115] p-5 space-y-4 shadow-xl animate-pulse">
              <div className="h-6 bg-zinc-800/80 rounded-lg w-2/3" />
              <div className="h-10 bg-zinc-800/60 rounded-lg" />
              <div className="space-y-3">
                <div className="h-3.5 bg-zinc-800/50 rounded w-full" />
                <div className="h-3.5 bg-zinc-800/50 rounded w-5/6" />
                <div className="h-3.5 bg-zinc-800/50 rounded w-4/6" />
              </div>
              <div className="h-8 bg-zinc-800/60 rounded-lg" />
            </div>
          ) : result ? (
            <div className="rounded-xl border border-zinc-800 bg-[#0F1115] shadow-xl overflow-hidden animate-scale-in space-y-4 p-5">
              {/* Result Header */}
              <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-zinc-900 border border-zinc-800">
                    {getSentimentIcon(result.sentiment)}
                  </div>
                  <div>
                    <Badge
                      variant={
                        result.sentiment === 'POSITIVE'
                          ? 'emerald'
                          : result.sentiment === 'NEGATIVE'
                          ? 'rose'
                          : 'amber'
                      }
                      size="md"
                    >
                      {result.sentiment}
                    </Badge>
                    <p className="text-[10px] text-zinc-400 mt-1 font-mono">
                      {getConfidenceLabel(result.confidence).label} confidence
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-zinc-100 font-mono tabular-nums leading-none">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-zinc-500 mt-0.5 font-mono">confidence</p>
                </div>
              </div>

              {/* Probability Distribution */}
              <div className="space-y-2.5">
                <p className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wider font-mono">
                  Probability Distribution
                </p>
                <ProbabilityBar
                  label="Positive"
                  percentage={result.probabilities.positive}
                  colorClass="bg-emerald-500"
                />
                <ProbabilityBar
                  label="Neutral"
                  percentage={result.probabilities.neutral}
                  colorClass="bg-amber-500"
                />
                <ProbabilityBar
                  label="Negative"
                  percentage={result.probabilities.negative}
                  colorClass="bg-rose-500"
                />
              </div>

              {/* Inference Metadata */}
              <div className="grid grid-cols-2 gap-2 pt-1">
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-0.5">
                  <div className="flex items-center gap-1 text-[10px] text-zinc-500 font-mono">
                    <Clock className="w-3 h-3" /> Latency
                  </div>
                  <p className="text-xs font-bold text-zinc-200 font-mono tabular-nums">
                    {result.latency_ms} ms
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-0.5">
                  <div className="flex items-center gap-1 text-[10px] text-zinc-500 font-mono">
                    <Cpu className="w-3 h-3" /> Engine
                  </div>
                  <p className="text-xs font-bold text-zinc-200 font-mono truncate">
                    TF Bi-LSTM
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-0.5">
                  <div className="flex items-center gap-1 text-[10px] text-zinc-500 font-mono">
                    <Activity className="w-3 h-3" /> Model Health
                  </div>
                  <p className="text-xs font-bold text-emerald-400 font-mono">Nominal</p>
                </div>
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-0.5">
                  <div className="flex items-center gap-1 text-[10px] text-zinc-500 font-mono">
                    <Calendar className="w-3 h-3" /> Timestamp
                  </div>
                  <p className="text-[11px] font-mono tabular-nums text-zinc-300 truncate">
                    {result.created_at
                      ? new Date(result.created_at).toLocaleTimeString()
                      : new Date().toLocaleTimeString()}
                  </p>
                </div>
              </div>

              {/* Target text preview */}
              <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800">
                <div className="flex items-center gap-1 text-[10px] text-zinc-500 mb-1 font-mono">
                  <Target className="w-3 h-3" /> Input Text Preview
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed line-clamp-2 font-sans">
                  {result.text_content}
                </p>
              </div>

              {/* Copy & Export Actions */}
              <div className="flex gap-2 pt-1">
                <Button
                  onClick={handleCopyResult}
                  variant="secondary"
                  size="sm"
                  className="flex-1"
                >
                  {copied ? (
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <Copy className="w-3.5 h-3.5 text-zinc-400" />
                  )}
                  <span>{copied ? 'Copied!' : 'Copy'}</span>
                </Button>
                <Button
                  onClick={handleExportJson}
                  variant="secondary"
                  size="sm"
                  className="flex-1"
                >
                  <Download className="w-3.5 h-3.5 text-zinc-400" />
                  <span>Export JSON</span>
                </Button>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="rounded-xl border border-zinc-800 border-dashed bg-[#0F1115]/60 p-8 flex flex-col items-center justify-center text-center space-y-3 min-h-[360px]">
              <div className="w-10 h-10 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center">
                <FileText className="w-5 h-5 text-zinc-500" />
              </div>
              <div>
                <p className="text-sm font-semibold text-zinc-200">Ready to analyze</p>
                <p className="text-xs text-zinc-400 mt-1 max-w-[200px] leading-relaxed">
                  Enter text or pick a sample review, then press Analyze.
                </p>
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-zinc-500 font-mono">
                <ChevronRight className="w-3 h-3" />
                <kbd className="px-1.5 py-0.5 bg-zinc-900 border border-zinc-800 rounded font-mono text-zinc-400">
                  Ctrl+Enter
                </kbd>{' '}
                to analyze
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Explainability Panel — full width below */}
      {result && result.explainability && result.explainability.length > 0 && (
        <Card className="p-5 shadow-xl animate-scale-in">
          <ExplainabilityTags
            tokens={result.explainability}
            sentiment={result.sentiment}
            confidence={result.confidence}
          />
        </Card>
      )}
    </div>
  );
};
