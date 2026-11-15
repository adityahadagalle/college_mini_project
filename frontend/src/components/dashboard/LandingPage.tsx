import React from 'react';
import {
  Zap,
  FileSpreadsheet,
  ArrowRight,
  Split,
  Heart,
  Boxes,
  ShieldCheck,
  Layers,
  BarChart3,
  CheckCircle2,
} from 'lucide-react';
import { Button } from '../common/Button';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';

interface LandingPageProps {
  onGetStarted: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted }) => {
  return (
    <div className="space-y-16 max-w-5xl mx-auto py-10">
      {/* Hero Section */}
      <div className="text-center space-y-5 relative">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#141720] border border-white/[0.08] shadow-[inset_0_1px_0_0_rgba(255,255,255,0.06)] text-zinc-300 text-xs font-mono">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span>4 Neural Engines Online • CPU & GPU Guarded</span>
        </div>

        <h1 className="text-3xl sm:text-5xl font-bold text-zinc-100 tracking-tight leading-[1.15]">
          Sentiment Intelligence Platform
        </h1>

        <p className="text-zinc-400 text-sm sm:text-base max-w-2xl mx-auto leading-relaxed font-sans">
          Production-grade customer feedback analytics combining calibrated document sentiment, span-level aspect extraction (ABSA), 28-label emotion taxonomy, and automatic semantic topic discovery.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-3">
          <Button onClick={onGetStarted} variant="primary" size="lg" icon={ArrowRight}>
            Launch Review Intelligence
          </Button>
        </div>

        {/* Feature Pills */}
        <div className="flex flex-wrap items-center justify-center gap-2 pt-4 text-xs">
          <Badge variant="zinc">Sub-300ms SLA</Badge>
          <Badge variant="emerald">DeBERTa-v3 ABSA</Badge>
          <Badge variant="zinc">GoEmotions Multi-Label</Badge>
          <Badge variant="zinc">UMAP + HDBSCAN Topics</Badge>
          <Badge variant="zinc">FastAPI Async Architecture</Badge>
        </div>
      </div>

      {/* 4 Engine Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card hoverable className="p-5 space-y-3">
          <div className="w-9 h-9 rounded-xl bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Zap className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-zinc-100">Document Sentiment</h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Overall 3-class polarity with probability calibration and gradient token attribution explainability.
          </p>
        </Card>

        <Card hoverable className="p-5 space-y-3">
          <div className="w-9 h-9 rounded-xl bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Split className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-zinc-100">Aspect-Based (ABSA)</h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Stage 1 ATE token extraction and Stage 2 ALSC cross-encoder polarity conditioned on explicit review spans.
          </p>
        </Card>

        <Card hoverable className="p-5 space-y-3">
          <div className="w-9 h-9 rounded-xl bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Heart className="w-4 h-4 text-rose-400" />
          </div>
          <h3 className="text-sm font-semibold text-zinc-100">Emotion & CX Rollup</h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Multi-label 28-class GoEmotions detection with calibrated thresholds rolled into 6 CX categories.
          </p>
        </Card>

        <Card hoverable className="p-5 space-y-3">
          <div className="w-9 h-9 rounded-xl bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Boxes className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-zinc-100">Topic Discovery</h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Sentence-Transformers + UMAP + HDBSCAN density clustering with centroid projection and c-TF-IDF labels.
          </p>
        </Card>
      </div>

      {/* Operational Highlights */}
      <div className="p-6 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 rounded-xl border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] backdrop-blur-md grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>Unified Workflow</span>
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Analyze reviews across all four capabilities simultaneously in one single roundtrip with transactional consistency.
          </p>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
            <FileSpreadsheet className="w-4 h-4 text-zinc-300" />
            <span>Asynchronous Batches</span>
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Stream review files via background batch jobs with live progress bars, pause/resume, and full CSV exports.
          </p>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
            <BarChart3 className="w-4 h-4 text-zinc-300" />
            <span>Cross-Feature Analytics</span>
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Correlate topics with aspect sentiments, emotion distributions, and statistical sample confidence intervals.
          </p>
        </div>
      </div>
    </div>
  );
};
