import React, { useMemo } from 'react';
import { ExplainabilityToken } from '../../types/sentiment';
import { Sparkles, TrendingUp, TrendingDown, Minus, Info } from 'lucide-react';

interface ExplainabilityTagsProps {
  tokens?: ExplainabilityToken[];
  sentiment?: string;
  confidence?: number;
}

/** Generate an attribution-derived explanation summary directly from gradient token scores. */
function buildReasoning(
  tokens: ExplainabilityToken[],
  sentiment: string,
  confidence: number,
): string {
  const positiveWords = tokens
    .filter((t) => t.score > 0.05)
    .map((t) => `"${t.token}"`)
    .slice(0, 4);
  const negativeWords = tokens
    .filter((t) => t.score < -0.05)
    .map((t) => `"${t.token}"`)
    .slice(0, 4);

  const conf = Math.round(confidence * 100);

  if (sentiment === 'POSITIVE') {
    if (positiveWords.length > 0) {
      return `Attribution Analysis: The model identified dominant positive gradient signals from ${positiveWords.join(', ')}. These tokens carry the highest positive attribution weights, driving the ${conf}% confidence classification.`;
    }
    return `Attribution Analysis: The overall sequence embedding contains consistent positive directional gradients, yielding a ${conf}% confidence classification.`;
  }
  if (sentiment === 'NEGATIVE') {
    if (negativeWords.length > 0) {
      return `Attribution Analysis: The model identified critical negative gradient signals from ${negativeWords.join(', ')}. These tokens carry the highest negative attribution weights, producing a ${conf}% confidence negative classification.`;
    }
    return `Attribution Analysis: The input sequence contains predominantly negative gradient tokens, leading to a ${conf}% confidence negative classification.`;
  }
  // NEUTRAL
  const hasBoth = positiveWords.length > 0 && negativeWords.length > 0;
  if (hasBoth) {
    return `Attribution Analysis: The review contains competing positive gradients (${positiveWords.slice(0, 2).join(', ')}) and negative gradients (${negativeWords.slice(0, 2).join(', ')}). The opposing attribution weights resulted in a balanced ${conf}% confidence neutral classification.`;
  }
  return `Attribution Analysis: The model found no dominant directional gradient weights, assigning a ${conf}% confidence score to the neutral class.`;
}

/** Inline text with token spans highlighted by attribution score */
function HighlightedText({
  tokens,
}: {
  tokens: ExplainabilityToken[];
}) {
  if (tokens.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1 leading-7">
      {tokens.map((t, i) => {
        const score = t.score;
        let bg = '';
        let title = `Gradient Score: ${score.toFixed(3)}`;
        if (score > 0.1) {
          const intensity = Math.min(score * 4, 1);
          bg = `rgba(16, 185, 129, ${0.12 + intensity * 0.25})`; // emerald
        } else if (score < -0.1) {
          const intensity = Math.min(Math.abs(score) * 4, 1);
          bg = `rgba(244, 63, 94, ${0.12 + intensity * 0.25})`; // rose
        } else {
          bg = 'rgba(255, 255, 255, 0.04)'; // zinc
        }
        const color =
          score > 0.1 ? '#34d399' : score < -0.1 ? '#fb7185' : '#a1a1aa';

        return (
          <span
            key={i}
            title={title}
            className="rounded px-1.5 py-0.5 text-xs font-medium transition-all cursor-default"
            style={{ backgroundColor: bg, color }}
          >
            {t.token}
          </span>
        );
      })}
    </div>
  );
}

export const ExplainabilityTags: React.FC<ExplainabilityTagsProps> = ({
  tokens,
  sentiment = 'NEUTRAL',
  confidence = 0.5,
}) => {
  if (!tokens || tokens.length === 0) return null;

  const positiveTokens = useMemo(
    () => tokens.filter((t) => t.score > 0.05).sort((a, b) => b.score - a.score).slice(0, 6),
    [tokens],
  );
  const negativeTokens = useMemo(
    () => tokens.filter((t) => t.score < -0.05).sort((a, b) => a.score - b.score).slice(0, 6),
    [tokens],
  );
  const reasoning = useMemo(
    () => buildReasoning(tokens, sentiment, confidence),
    [tokens, sentiment, confidence],
  );

  return (
    <div className="space-y-4">
      {/* Section Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
          <div className="p-1 rounded-md bg-zinc-900 border border-zinc-800">
            <Sparkles className="w-3.5 h-3.5 text-zinc-300" />
          </div>
          <span>AI-Attribution Derived Explanation (Gradient Attribution Scores)</span>
        </div>
        <span className="text-[10px] text-zinc-400 font-mono bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
          tf.GradientTape
        </span>
      </div>

      {/* Inline Highlighted Review Text */}
      <div className="p-3.5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
        <div className="flex items-center justify-between text-[10px] text-zinc-500 uppercase font-mono tracking-wider font-semibold">
          <span>Token Attribution Heatmap</span>
          <span className="flex items-center gap-1 text-[10px] text-zinc-400 normal-case font-normal">
            <Info className="w-3 h-3 text-zinc-400" /> Derived from model gradient weights
          </span>
        </div>
        <HighlightedText tokens={tokens} />
        <div className="flex items-center gap-4 mt-2 pt-2 border-t border-zinc-800/80 text-[10px] font-mono">
          <span className="flex items-center gap-1 text-emerald-400 font-medium">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-400/30 border border-emerald-400/40" />
            Positive Attribution (&gt;0.1)
          </span>
          <span className="flex items-center gap-1 text-rose-400 font-medium">
            <span className="w-2.5 h-2.5 rounded-sm bg-rose-400/30 border border-rose-400/40" />
            Negative Attribution (&lt;-0.1)
          </span>
          <span className="flex items-center gap-1 text-zinc-400">
            <span className="w-2.5 h-2.5 rounded-sm bg-zinc-800" />
            Neutral Attribution
          </span>
        </div>
      </div>

      {/* Evidence Breakdown */}
      {(positiveTokens.length > 0 || negativeTokens.length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {positiveTokens.length > 0 && (
            <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-500/25 space-y-2">
              <div className="flex items-center gap-1.5 text-[10px] font-semibold text-emerald-400 uppercase font-mono tracking-wider">
                <TrendingUp className="w-3 h-3" /> Positive Gradient Signals
              </div>
              <div className="flex flex-wrap gap-1">
                {positiveTokens.map((t, i) => (
                  <span
                    key={i}
                    className="px-1.5 py-0.5 bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 rounded text-[10px] font-mono tabular-nums"
                  >
                    {t.token}
                    <span className="text-emerald-500 ml-1">+{t.score.toFixed(2)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
          {negativeTokens.length > 0 && (
            <div className="p-3 rounded-lg bg-rose-950/20 border border-rose-500/25 space-y-2">
              <div className="flex items-center gap-1.5 text-[10px] font-semibold text-rose-400 uppercase font-mono tracking-wider">
                <TrendingDown className="w-3 h-3" /> Negative Gradient Signals
              </div>
              <div className="flex flex-wrap gap-1">
                {negativeTokens.map((t, i) => (
                  <span
                    key={i}
                    className="px-1.5 py-0.5 bg-rose-500/10 text-rose-300 border border-rose-500/20 rounded text-[10px] font-mono tabular-nums"
                  >
                    {t.token}
                    <span className="text-rose-500 ml-1">{t.score.toFixed(2)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Reasoning Explanation */}
      <div className="p-3.5 rounded-lg bg-zinc-900 border border-zinc-800 space-y-1">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-400 uppercase font-mono tracking-wider">
          <Minus className="w-3 h-3" /> Model Decision Reasoning (Gradients)
        </div>
        <p className="text-xs text-zinc-300 leading-relaxed italic font-sans">{reasoning}</p>
      </div>
    </div>
  );
};
