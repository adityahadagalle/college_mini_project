import React, { useState, useEffect } from 'react';
import {
  Boxes,
  Sparkles,
  RefreshCw,
  Edit2,
  GitMerge,
  Archive,
  Layers,
  Check,
  X,
  AlertCircle,
  HelpCircle,
  ChevronRight,
  TrendingUp,
  Cpu,
  Database,
  ArrowRight,
  ExternalLink,
  Search,
  Sliders,
  Award,
} from 'lucide-react';
import { apiService } from '../../services/api';
import {
  DiscoveredTopicItem,
  TopicComparisonResponse,
  TopicListResponse,
  TopicMetadataResponse,
  TopicReviewsResponse,
  TopicReviewAssignmentItem,
} from '../../types/sentiment';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { SkeletonLoader } from '../common/SkeletonLoader';
import { EmptyState } from '../common/EmptyState';

const SAMPLE_BENCHMARK_PROMPT = [
  "The battery life on this phone is absolutely stellar, easily lasts two full days.",
  "Battery drains completely within four hours of light usage, terrible performance.",
  "Fast charging takes less than 30 minutes to reach 80% charge, very impressive.",
  "The phone gets uncomfortably hot while charging via the fast charger.",
  "Battery health degraded to 85% after just six months of daily use.",
  "The sound quality is crisp and the bass response is deep without distortion.",
  "Speakers sound muddy and hollow at high volume, very disappointing audio.",
  "Active noise cancellation works wonders on public transport and flights.",
  "Microphone quality during phone calls is muffled according to callers.",
  "Bluetooth audio connection drops frequently when walking around the room.",
  "Customer support answered my ticket within ten minutes and issued a prompt replacement.",
  "Shipping was delayed by two weeks with zero tracking updates provided.",
  "Packaging arrived completely crushed and the product box was dented inside.",
  "Support team was completely unhelpful and refused to honor the warranty agreement.",
  "Delivery arrived two days ahead of schedule in pristine condition.",
  "The user interface is fluid, animations are smooth, and apps launch instantly.",
  "Frequent app crashes and freezing make the software nearly unusable on daily tasks.",
  "The latest software update introduced serious lag and stutter when scrolling.",
  "Clean operating system with zero bloatware pre-installed, clean and refreshing.",
  "Screen scratched easily on the first day despite claimed durability.",
];

export const TopicExplorer: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [runningDiscovery, setRunningDiscovery] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Data
  const [topicData, setTopicData] = useState<TopicListResponse | null>(null);
  const [metadata, setMetadata] = useState<TopicMetadataResponse | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string>('');

  // Selected topic for review drawer
  const [selectedTopic, setSelectedTopic] = useState<DiscoveredTopicItem | null>(null);
  const [reviewData, setReviewData] = useState<TopicReviewsResponse | null>(null);
  const [loadingReviews, setLoadingReviews] = useState<boolean>(false);
  const [currentPage, setCurrentPage] = useState<number>(1);

  // Edit Label State
  const [editingTopicId, setEditingTopicId] = useState<string | null>(null);
  const [editLabelText, setEditLabelText] = useState<string>('');

  // Merge Modal State
  const [mergeSourceTopic, setMergeSourceTopic] = useState<DiscoveredTopicItem | null>(null);
  const [mergeTargetId, setMergeTargetId] = useState<string>('');
  const [mergeNewLabel, setMergeNewLabel] = useState<string>('');

  // Comparison State
  const [showComparison, setShowComparison] = useState<boolean>(false);
  const [compareRunA, setCompareRunA] = useState<string>('');
  const [compareRunB, setCompareRunB] = useState<string>('');
  const [comparisonResult, setComparisonResult] = useState<TopicComparisonResponse | null>(null);
  const [loadingComparison, setLoadingComparison] = useState<boolean>(false);

  // Load initial data
  useEffect(() => {
    fetchInitialData();
  }, []);

  // Escape key listener for review drawer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (selectedTopic) setSelectedTopic(null);
        if (mergeSourceTopic) setMergeSourceTopic(null);
        if (showComparison) setShowComparison(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedTopic, mergeSourceTopic, showComparison]);

  const fetchInitialData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [topicsRes, metaRes] = await Promise.allSettled([
        apiService.getTopics(),
        apiService.getTopicMetadata(),
      ]);

      if (topicsRes.status === 'fulfilled') {
        setTopicData(topicsRes.value);
        if (topicsRes.value.run_id) {
          setSelectedRunId(topicsRes.value.run_id);
        }
      }

      if (metaRes.status === 'fulfilled') {
        setMetadata(metaRes.value);
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to load topic discovery data.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectRun = async (runId: string) => {
    setSelectedRunId(runId);
    setLoading(true);
    try {
      const res = await apiService.getTopics(runId);
      setTopicData(res);
      setSelectedTopic(null);
      setReviewData(null);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to switch discovery run.');
    } finally {
      setLoading(false);
    }
  };

  const handleTriggerDiscovery = async (useCustomCorpus: boolean = false) => {
    setRunningDiscovery(true);
    setError(null);
    setSuccessMsg(null);
    try {
      const payload = useCustomCorpus
        ? {
            source: 'custom_texts' as const,
            custom_texts: SAMPLE_BENCHMARK_PROMPT,
            min_cluster_size: 3,
            n_neighbors: 10,
            n_components: 5,
          }
        : {
            source: 'all_predictions' as const,
            min_cluster_size: 3,
            n_neighbors: 15,
            n_components: 5,
          };

      const res = await apiService.discoverTopics(payload, true);
      setTopicData(res);
      setSelectedRunId(res.run_id);
      setSuccessMsg(
        `Discovered ${res.num_topics} topics across ${res.corpus_size} reviews (${res.outlier_count} outliers isolated).`
      );
    } catch (err: any) {
      const detail = err?.response?.data?.detail || err.message || 'Discovery run failed.';
      setError(detail);
    } finally {
      setRunningDiscovery(false);
    }
  };

  const handleOpenReviews = async (topic: DiscoveredTopicItem, page: number = 1) => {
    setSelectedTopic(topic);
    setCurrentPage(page);
    setLoadingReviews(true);
    try {
      const res = await apiService.getTopicReviews(topic.id, page, 15);
      setReviewData(res);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to fetch topic reviews.');
    } finally {
      setLoadingReviews(false);
    }
  };

  const handleSaveRename = async (topicId: string) => {
    if (!editLabelText.trim()) return;
    try {
      const updated = await apiService.renameTopic(topicId, editLabelText.trim());
      if (topicData) {
        setTopicData({
          ...topicData,
          topics: topicData.topics.map((t) => (t.id === topicId ? updated : t)),
        });
      }
      setEditingTopicId(null);
      setSuccessMsg(`Topic renamed to "${updated.human_label}".`);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to rename topic.');
    }
  };

  const handleMergeSubmit = async () => {
    if (!mergeSourceTopic || !mergeTargetId) return;
    try {
      const updatedTarget = await apiService.mergeTopics(
        mergeSourceTopic.id,
        mergeTargetId,
        mergeNewLabel ? mergeNewLabel.trim() : undefined
      );
      if (topicData) {
        setTopicData({
          ...topicData,
          topics: topicData.topics
            .filter((t) => t.id !== mergeSourceTopic.id)
            .map((t) => (t.id === updatedTarget.id ? updatedTarget : t)),
        });
      }
      setMergeSourceTopic(null);
      setMergeTargetId('');
      setMergeNewLabel('');
      setSuccessMsg('Topics merged successfully.');
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to merge topics.');
    }
  };

  const handleArchive = async (topicId: string) => {
    if (!confirm('Are you sure you want to archive this topic?')) return;
    try {
      await apiService.archiveTopic(topicId);
      if (topicData) {
        setTopicData({
          ...topicData,
          topics: topicData.topics.filter((t) => t.id !== topicId),
        });
      }
      setSuccessMsg('Topic archived from active view.');
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to archive topic.');
    }
  };

  const handleRunComparison = async () => {
    if (!compareRunA || !compareRunB) return;
    setLoadingComparison(true);
    setComparisonResult(null);
    try {
      const res = await apiService.compareTopicRuns(compareRunA, compareRunB);
      setComparisonResult(res);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to compare runs.');
    } finally {
      setLoadingComparison(false);
    }
  };

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Boxes className="w-4 h-4 text-zinc-300" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">
              Automatic Topic Discovery
            </h1>
            <Badge variant="zinc" size="sm">all-MiniLM-L6-v2 + HDBSCAN</Badge>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {topicData?.all_runs && topicData.all_runs.length > 1 && (
            <select
              value={selectedRunId}
              onChange={(e) => handleSelectRun(e.target.value)}
              className="bg-[#07080C] text-xs text-zinc-200 border border-white/[0.08] shadow-inner rounded-lg px-2.5 py-1.5 outline-none focus:border-zinc-400 font-mono cursor-pointer"
            >
              {topicData.all_runs.map((r, idx) => (
                <option key={r.id} value={r.id}>
                  Run #{idx + 1} ({r.num_topics} topics, {r.corpus_size} reviews)
                </option>
              ))}
            </select>
          )}

          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              if (topicData?.all_runs && topicData.all_runs.length >= 2) {
                setCompareRunA(topicData.all_runs[1].id);
                setCompareRunB(topicData.all_runs[0].id);
              }
              setShowComparison(!showComparison);
            }}
          >
            <Sliders className="w-3.5 h-3.5 mr-1.5 text-zinc-400" />
            Compare Runs
          </Button>

          <Button
            variant="secondary"
            size="sm"
            disabled={runningDiscovery}
            onClick={() => handleTriggerDiscovery(true)}
          >
            <Database className="w-3.5 h-3.5 mr-1.5 text-zinc-400" />
            Benchmark Corpus
          </Button>

          <Button
            variant="primary"
            size="sm"
            disabled={runningDiscovery}
            onClick={() => handleTriggerDiscovery(false)}
          >
            <Sparkles className="w-3.5 h-3.5 mr-1.5" />
            {runningDiscovery ? 'Clustering...' : 'Discover Topics'}
          </Button>
        </div>
      </div>

      {/* Notifications */}
      {error && (
        <div className="bg-rose-950/30 border border-rose-800/60 rounded-xl p-3.5 flex items-center justify-between text-rose-300 text-xs">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="hover:text-rose-200 p-1">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {successMsg && (
        <div className="bg-emerald-950/30 border border-emerald-800/60 rounded-xl p-3.5 flex items-center justify-between text-emerald-300 text-xs">
          <div className="flex items-center gap-2">
            <Check className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>{successMsg}</span>
          </div>
          <button onClick={() => setSuccessMsg(null)} className="hover:text-emerald-200 p-1">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* KPI Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <Card className="p-4 space-y-1">
          <div className="text-zinc-400 text-[11px] font-semibold uppercase tracking-wider">
            Topics Discovered
          </div>
          <div className="text-2xl font-bold font-mono text-zinc-100 flex items-center justify-between tabular-nums">
            {topicData?.num_topics ?? 0}
            <Boxes className="w-4 h-4 text-zinc-500" />
          </div>
          <div className="text-[10px] text-zinc-500">Excluding outlier cluster</div>
        </Card>

        <Card className="p-4 space-y-1">
          <div className="text-zinc-400 text-[11px] font-semibold uppercase tracking-wider">
            Corpus Size
          </div>
          <div className="text-2xl font-bold font-mono text-zinc-100 flex items-center justify-between tabular-nums">
            {topicData?.corpus_size ?? 0}
            <Database className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-[10px] text-zinc-500">Min threshold: 15 unique reviews</div>
        </Card>

        <Card className="p-4 space-y-1">
          <div className="text-zinc-400 text-[11px] font-semibold uppercase tracking-wider">
            Outliers Isolated
          </div>
          <div className="text-2xl font-bold font-mono text-zinc-100 flex items-center justify-between tabular-nums">
            {topicData?.outlier_count ?? 0}
            <span className="text-xs font-mono text-amber-400">
              {topicData?.metrics?.outlier_ratio
                ? `${(topicData.metrics.outlier_ratio * 100).toFixed(1)}%`
                : '0%'}
            </span>
          </div>
          <div className="text-[10px] text-zinc-500">HDBSCAN unclustered buffer (-1)</div>
        </Card>

        <Card className="p-4 space-y-1">
          <div className="text-zinc-400 text-[11px] font-semibold uppercase tracking-wider">
            Intra-Cluster Cohesion
          </div>
          <div className="text-2xl font-bold font-mono text-zinc-100 flex items-center justify-between tabular-nums">
            {topicData?.metrics?.mean_intra_cluster_cohesion
              ? topicData.metrics.mean_intra_cluster_cohesion.toFixed(3)
              : 'N/A'}
            <TrendingUp className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-[10px] text-zinc-500">Mean cosine sim to centroid</div>
        </Card>

        <Card className="p-4 space-y-1">
          <div className="text-zinc-400 text-[11px] font-semibold uppercase tracking-wider">
            Keyword Diversity
          </div>
          <div className="text-2xl font-bold font-mono text-zinc-100 flex items-center justify-between tabular-nums">
            {topicData?.metrics?.keyword_diversity_ratio
              ? `${(topicData.metrics.keyword_diversity_ratio * 100).toFixed(0)}%`
              : 'N/A'}
            <Sparkles className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-[10px] text-zinc-500">Distinct c-TF-IDF tokens</div>
        </Card>
      </div>

      {/* Comparison Drawer / Panel */}
      {showComparison && (
        <Card className="p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-zinc-400" />
              <h3 className="text-sm font-semibold text-zinc-200">Cross-Run Drift & Emerging Themes</h3>
            </div>
            <button onClick={() => setShowComparison(false)} className="text-zinc-400 hover:text-white p-1">
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div>
              <label className="text-xs text-zinc-400 font-medium block mb-1">Baseline Run (A)</label>
              <select
                value={compareRunA}
                onChange={(e) => setCompareRunA(e.target.value)}
                className="w-full bg-[#07080C] border border-white/[0.08] shadow-inner text-xs text-zinc-200 rounded-lg px-3 py-2 outline-none focus:border-zinc-400 font-mono"
              >
                <option value="">Select Baseline Run</option>
                {topicData?.all_runs?.map((r, i) => (
                  <option key={r.id} value={r.id}>
                    Run #{i + 1} ({r.id.slice(0, 8)}...) - {r.num_topics} topics
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-zinc-400 font-medium block mb-1">Comparison Run (B)</label>
              <select
                value={compareRunB}
                onChange={(e) => setCompareRunB(e.target.value)}
                className="w-full bg-[#07080C] border border-white/[0.08] shadow-inner text-xs text-zinc-200 rounded-lg px-3 py-2 outline-none focus:border-zinc-400 font-mono"
              >
                <option value="">Select Comparison Run</option>
                {topicData?.all_runs?.map((r, i) => (
                  <option key={r.id} value={r.id}>
                    Run #{i + 1} ({r.id.slice(0, 8)}...) - {r.num_topics} topics
                  </option>
                ))}
              </select>
            </div>

            <Button
              variant="primary"
              disabled={!compareRunA || !compareRunB || loadingComparison}
              onClick={handleRunComparison}
            >
              {loadingComparison ? 'Comparing...' : 'Compute Drift & Emerging Themes'}
            </Button>
          </div>

          {comparisonResult && (
            <div className="space-y-4 pt-4 border-t border-zinc-800">
              {/* Candidate Emerging Themes Banner */}
              {comparisonResult.emerging_topics_in_b.length > 0 ? (
                <div className="bg-amber-950/20 border border-amber-800/40 rounded-xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-amber-400 text-xs font-semibold">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>Candidate Emerging Themes Detected ({comparisonResult.emerging_topics_in_b.length})</span>
                  </div>
                  <p className="text-xs text-zinc-400">
                    {comparisonResult.emerging_topics_in_b[0].uncertainty_note}
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                    {comparisonResult.emerging_topics_in_b.map((et) => (
                      <div key={et.topic_id} className="bg-[#161920] p-3 rounded-lg border border-zinc-800">
                        <div className="flex justify-between items-start">
                          <span className="text-xs font-semibold text-zinc-100">{et.topic_name}</span>
                          <Badge variant="amber" size="sm" className="tabular-nums">
                            Max Sim: {(et.max_similarity_to_prior_run * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <div className="text-[11px] text-zinc-500 mt-1 font-mono tabular-nums">{et.review_count} reviews</div>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {et.keywords.slice(0, 4).map((k) => (
                            <span key={k.word} className="text-[10px] bg-[#09090B] text-zinc-300 px-1.5 py-0.5 rounded border border-zinc-800">
                              {k.word}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-xs text-zinc-400 bg-[#09090B] p-3 rounded-lg border border-zinc-800">
                  Zero candidate emerging themes with similarity &lt; 0.45 detected. All clusters align with baseline.
                </div>
              )}

              {/* Matched Topics Table */}
              <div>
                <h4 className="text-xs font-semibold uppercase text-zinc-400 mb-2">
                  Matched Stable Topics (Cosine Sim &ge; 0.45)
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {comparisonResult.matched_topics.map((m, idx) => (
                    <div key={idx} className="flex items-center justify-between bg-[#161920] p-2.5 rounded-lg border border-zinc-800 text-xs">
                      <span className="text-zinc-300">{m.topic_a_name}</span>
                      <ArrowRight className="w-3.5 h-3.5 text-zinc-500" />
                      <span className="text-zinc-100 font-medium">{m.topic_b_name}</span>
                      <Badge variant="positive" size="sm" className="tabular-nums">
                        {(m.similarity * 100).toFixed(0)}% match
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Discovered Topics Grid */}
      {loading ? (
        <SkeletonLoader variant="table" count={4} />
      ) : !topicData || topicData.topics.length === 0 ? (
        <EmptyState
          title="No Topics Discovered Yet"
          description="Click 'Discover Topics' or 'Benchmark Corpus' to run the Sentence-Transformers + UMAP + HDBSCAN discovery pipeline."
          icon={<Boxes className="w-6 h-6 text-zinc-400" />}
          actionLabel="Run Topic Discovery"
          onAction={() => handleTriggerDiscovery(true)}
        />
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
              Discovered Topic Clusters ({topicData.topics.length})
            </h2>
            <div className="text-xs text-zinc-400 font-mono tabular-nums">
              Run ID: <code className="text-zinc-200">{topicData.run_id.slice(0, 8)}</code>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {topicData.topics.map((topic) => {
              const isOutlier = topic.topic_index === -1;
              const isEditing = editingTopicId === topic.id;

              return (
                <Card
                  key={topic.id}
                  className={`p-5 transition-all duration-200 flex flex-col justify-between ${
                    isOutlier
                      ? 'border-amber-800/40 hover:border-amber-700/60'
                      : 'hover:border-zinc-500'
                  }`}
                >
                  <div>
                    {/* Header: Index + Label + Curate Controls */}
                    <div className="flex items-start justify-between gap-2 mb-3">
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        <Badge
                          variant={isOutlier ? 'amber' : 'zinc'}
                          size="sm"
                          className="font-mono tabular-nums"
                        >
                          {isOutlier ? 'Outlier' : `#${topic.topic_index + 1}`}
                        </Badge>

                        {isEditing ? (
                          <div className="flex items-center gap-1 flex-1">
                            <input
                              type="text"
                              value={editLabelText}
                              onChange={(e) => setEditLabelText(e.target.value)}
                              className="bg-[#09090B] text-xs text-white px-2 py-1 rounded border border-zinc-600 outline-none w-full"
                              autoFocus
                            />
                            <button
                              onClick={() => handleSaveRename(topic.id)}
                              className="p-1 text-emerald-400 hover:bg-zinc-800 rounded"
                            >
                              <Check className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={() => setEditingTopicId(null)}
                              className="p-1 text-zinc-400 hover:bg-zinc-800 rounded"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ) : (
                          <span className="text-sm font-semibold text-zinc-100 tracking-tight leading-snug line-clamp-1">
                            {topic.display_label}
                          </span>
                        )}
                      </div>

                      {/* Action Menu Buttons */}
                      {!isEditing && (
                        <div className="flex items-center gap-1 text-zinc-400">
                          <button
                            title="Rename Topic"
                            onClick={() => {
                              setEditingTopicId(topic.id);
                              setEditLabelText(topic.human_label || topic.auto_label);
                            }}
                            className="p-1 hover:text-zinc-200 hover:bg-zinc-800 rounded"
                          >
                            <Edit2 className="w-3.5 h-3.5" />
                          </button>
                          {!isOutlier && (
                            <>
                              <button
                                title="Merge with another topic"
                                onClick={() => setMergeSourceTopic(topic)}
                                className="p-1 hover:text-zinc-200 hover:bg-zinc-800 rounded"
                              >
                                <GitMerge className="w-3.5 h-3.5" />
                              </button>
                              <button
                                title="Archive topic"
                                onClick={() => handleArchive(topic.id)}
                                className="p-1 hover:text-rose-400 hover:bg-zinc-800 rounded"
                              >
                                <Archive className="w-3.5 h-3.5" />
                              </button>
                            </>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Keywords Cloud */}
                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {topic.keywords.slice(0, 6).map((k) => (
                        <span
                          key={k.word}
                          className="text-[11px] bg-[#09090B] text-zinc-300 px-2 py-0.5 rounded border border-zinc-800 font-mono"
                        >
                          {k.word}
                        </span>
                      ))}
                    </div>

                    {/* Cohesion & Sentiment distribution */}
                    <div className="space-y-2 mb-4 pt-3 border-t border-zinc-800 text-xs">
                      <div className="flex justify-between items-center text-zinc-400">
                        <span>Intra-Cluster Cohesion:</span>
                        <span className="font-mono text-zinc-200 tabular-nums">
                          {topic.cohesion_score ? topic.cohesion_score.toFixed(3) : 'N/A'}
                        </span>
                      </div>

                      {topic.sentiment_breakdown && Object.keys(topic.sentiment_breakdown).length > 0 && (
                        <div>
                          <div className="flex justify-between text-[10px] text-zinc-400 mb-1 font-mono tabular-nums">
                            <span>Sentiment Distribution</span>
                            <span>
                              {Math.round((topic.sentiment_breakdown['POSITIVE'] || 0) * 100)}% Pos /{' '}
                              {Math.round((topic.sentiment_breakdown['NEGATIVE'] || 0) * 100)}% Neg
                            </span>
                          </div>
                          <div className="h-1.5 w-full bg-[#09090B] rounded-full overflow-hidden flex border border-zinc-800">
                            <div
                              className="bg-emerald-500 h-full"
                              style={{ width: `${(topic.sentiment_breakdown['POSITIVE'] || 0) * 100}%` }}
                            />
                            <div
                              className="bg-zinc-600 h-full"
                              style={{ width: `${(topic.sentiment_breakdown['NEUTRAL'] || 0) * 100}%` }}
                            />
                            <div
                              className="bg-rose-500 h-full"
                              style={{ width: `${(topic.sentiment_breakdown['NEGATIVE'] || 0) * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Footer Card Action */}
                  <Button
                    variant="secondary"
                    size="sm"
                    className="w-full justify-between text-xs"
                    onClick={() => handleOpenReviews(topic, 1)}
                  >
                    <span className="tabular-nums">Inspect {topic.review_count} Reviews</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </Button>
                </Card>
              );
            })}
          </div>
        </div>
      )}

      {/* Review Inspector Drawer */}
      {selectedTopic && (
        <div className="fixed inset-0 bg-[#09090B]/80 backdrop-blur-xs z-50 flex justify-end">
          <div className="w-full max-w-2xl bg-[#0F1115] border-l border-zinc-800 h-full p-6 flex flex-col justify-between overflow-y-auto">
            <div>
              {/* Drawer Header */}
              <div className="flex items-start justify-between pb-4 border-b border-zinc-800 mb-6">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant={selectedTopic.topic_index === -1 ? 'amber' : 'zinc'} size="sm">
                      {selectedTopic.topic_index === -1 ? 'Outlier Cluster' : `Cluster #${selectedTopic.topic_index + 1}`}
                    </Badge>
                    <h3 className="text-base font-bold text-zinc-100">{selectedTopic.display_label}</h3>
                  </div>
                  <p className="text-xs text-zinc-400">
                    Showing reviews assigned to this cluster. Top representative reviews rank closest to centroid.
                  </p>
                </div>
                <button
                  onClick={() => setSelectedTopic(null)}
                  className="p-1.5 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Reviews List */}
              {loadingReviews ? (
                <SkeletonLoader variant="table" count={3} />
              ) : !reviewData || reviewData.reviews.length === 0 ? (
                <div className="text-center py-12 text-zinc-500 text-sm">
                  No review assignments found for this topic.
                </div>
              ) : (
                <div className="space-y-3">
                  {reviewData.reviews.map((rev) => (
                    <div
                      key={rev.id}
                      className={`p-4 rounded-xl border transition-all ${
                        rev.is_representative
                          ? 'bg-[#161920] border-zinc-500 shadow-xs'
                          : 'bg-[#09090B] border-zinc-800'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2">
                          {rev.is_representative && (
                            <Badge variant="zinc" size="sm">
                              <Award className="w-3 h-3 mr-1" />
                              Representative Example
                            </Badge>
                          )}
                          {rev.sentiment_label && (
                            <Badge
                              variant={
                                rev.sentiment_label === 'POSITIVE'
                                  ? 'positive'
                                  : rev.sentiment_label === 'NEGATIVE'
                                  ? 'negative'
                                  : 'neutral'
                              }
                              size="sm"
                            >
                              {rev.sentiment_label}
                            </Badge>
                          )}
                        </div>

                        {rev.distance_to_centroid !== null && (
                          <span className="text-[11px] font-mono text-zinc-400 tabular-nums">
                            Dist: {rev.distance_to_centroid.toFixed(3)}
                          </span>
                        )}
                      </div>

                      <p className="text-sm text-zinc-200 leading-relaxed font-sans">
                        "{rev.review_text}"
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Pagination Controls */}
            {reviewData && reviewData.total_reviews > reviewData.page_size && (
              <div className="pt-4 border-t border-zinc-800 flex items-center justify-between text-xs text-zinc-400 tabular-nums">
                <span>
                  Showing page {reviewData.page} of {Math.ceil(reviewData.total_reviews / reviewData.page_size)} ({reviewData.total_reviews} total reviews)
                </span>
                <div className="flex gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={reviewData.page <= 1}
                    onClick={() => handleOpenReviews(selectedTopic, reviewData.page - 1)}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={
                      reviewData.page >= Math.ceil(reviewData.total_reviews / reviewData.page_size)
                    }
                    onClick={() => handleOpenReviews(selectedTopic, reviewData.page + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Merge Modal */}
      {mergeSourceTopic && (
        <div className="fixed inset-0 bg-[#09090B]/80 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <Card className="max-w-md w-full p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <GitMerge className="w-4 h-4 text-zinc-300" />
                <h3 className="text-sm font-bold text-zinc-100">Merge Discovered Topics</h3>
              </div>
              <button onClick={() => setMergeSourceTopic(null)} className="text-zinc-400 hover:text-white p-1">
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-zinc-400 leading-relaxed">
              Merge <strong className="text-zinc-200">"{mergeSourceTopic.display_label}"</strong> into another topic. All reviews will be reassigned and the source topic archived.
            </p>

            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-300 font-medium block mb-1">
                  Select Target Topic
                </label>
                <select
                  value={mergeTargetId}
                  onChange={(e) => setMergeTargetId(e.target.value)}
                  className="w-full bg-[#09090B] border border-zinc-800 text-xs text-zinc-200 rounded-lg px-3 py-2 outline-none focus:border-zinc-500 font-mono"
                >
                  <option value="">-- Choose destination topic --</option>
                  {topicData?.topics
                    .filter((t) => t.id !== mergeSourceTopic.id && t.topic_index !== -1)
                    .map((t) => (
                      <option key={t.id} value={t.id}>
                        #{t.topic_index + 1}: {t.display_label} ({t.review_count} reviews)
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <label className="text-xs text-zinc-300 font-medium block mb-1">
                  New Topic Label (Optional)
                </label>
                <input
                  type="text"
                  placeholder="e.g. Battery Life & Fast Charging"
                  value={mergeNewLabel}
                  onChange={(e) => setMergeNewLabel(e.target.value)}
                  className="w-full bg-[#09090B] border border-zinc-800 text-xs text-zinc-200 rounded-lg px-3 py-2 outline-none focus:border-zinc-500"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-zinc-800">
              <Button variant="secondary" size="sm" onClick={() => setMergeSourceTopic(null)}>
                Cancel
              </Button>
              <Button variant="primary" size="sm" disabled={!mergeTargetId} onClick={handleMergeSubmit}>
                Confirm Merge
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};
