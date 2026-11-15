export interface User {
  id: string;
  email: string;
  full_name: string;
  role?: string;
  is_active: boolean;
  created_at: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  user: User;
}

export interface UserSettings {
  confidence_threshold: number;
  default_export_format: string;
  theme: string;
}

export interface ExplainabilityToken {
  token: string;
  score: number;
}

export interface PredictResponse {
  id: string;
  text_content: string;
  sentiment: 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';
  confidence: number;
  probabilities: {
    positive: number;
    neutral: number;
    negative: number;
  };
  latency_ms: number;
  explainability?: ExplainabilityToken[];
  created_at: string;
}

export interface BatchJobResponse {
  job_id: string;
  filename: string;
  total_rows: number;
  processed_rows: number;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  result_file_path?: string;
  created_at: string;
  completed_at?: string;
}

export interface AnalyticsOverview {
  total_predictions: number;
  positive_count: number;
  neutral_count: number;
  negative_count: number;
  average_confidence: number;
  average_latency_ms: number;
  sentiment_ratio: {
    positive: number;
    neutral: number;
    negative: number;
  };
}

export interface TrendDataPoint {
  date: string;
  total: number;
  positive: number;
  neutral: number;
  negative: number;
  avg_confidence: number;
}

export interface AnalyticsTrendsResponse {
  period: 'daily' | 'weekly' | 'monthly' | 'yearly';
  trends: TrendDataPoint[];
}

export interface ChartDataset {
  pie_chart: {
    labels: string[];
    values: number[];
  };
  bar_chart: {
    labels: string[];
    values: number[];
  };
  heatmap: {
    day: string;
    hour: number;
    count: number;
  }[];
}

export interface ModelInfoResponse {
  model_version: string;
  architecture: string;
  parameter_count: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  training_date: string;
}

export interface SystemHealthResponse {
  status: string;
  cpu_percent: number;
  memory_mb: number;
  memory_percent: number;
  uptime_seconds: number;
  model_loaded: boolean;
}

export interface ModelVersionItem {
  id: string;
  version: string;
  architecture: string;
  framework: string;
  status: string;
  is_active: boolean;
  artifact_path: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  parameters_count: number;
  description?: string;
  created_at: string;
}

export interface DriftMetricData {
  id: string;
  model_version: string;
  drift_score: number;
  drift_detected: boolean;
  confidence_drift: number;
  class_distribution: Record<string, number>;
  sample_size: number;
  recorded_at: string;
}

export interface ExperimentRunItem {
  id: string;
  experiment_name: string;
  run_name: string;
  status: string;
  hyperparameters: Record<string, any>;
  metrics: Record<string, any>;
  accuracy: number;
  val_loss: number;
  f1_score: number;
  roc_auc: number;
  training_duration_seconds: number;
  created_at: string;
}

export interface AlertRecordItem {
  id: string;
  alert_type: string;
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
  message: string;
  is_active: boolean;
  created_at: string;
}

export interface MlOpsSystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  memory_used_mb: number;
  gpu_percent?: number;
  inference_latency_p95_ms: number;
  request_rate_per_sec: number;
  error_rate_percent: number;
  batch_throughput_per_sec: number;
  queue_depth: number;
  model_loading_time_ms: number;
  active_users: number;
}

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastNotification {
  id: string;
  type: ToastType;
  title: string;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
}

// ---------------------------------------------------------------------------
// Aspect-Based Sentiment Analysis (ABSA) Types
// ---------------------------------------------------------------------------

export interface AspectResult {
  aspect: string;
  aspect_category?: string | null;
  is_implicit: boolean;
  start_char: number;
  end_char: number;
  sentiment: 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';
  confidence: number;
  probabilities: {
    POSITIVE: number;
    NEUTRAL: number;
    NEGATIVE: number;
  };
  evidence_text: string;
  model_name: string;
  model_version: string;
  created_at: string;
}

export interface ABSAResponse {
  text: string;
  aspect_count: number;
  aspects: AspectResult[];
  latency_ms: number;
  overall_sentiment_preview: 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';
}

export interface ABSABatchResponse {
  total_samples: number;
  total_aspects: number;
  latency_ms: number;
  results: ABSAResponse[];
}

export interface AspectFrequencyItem {
  aspect: string;
  count: number;
  positive_count: number;
  neutral_count: number;
  negative_count: number;
  net_sentiment_score: number;
}

export interface AspectAnalyticsResponse {
  total_aspects: number;
  unique_aspects_count: number;
  sentiment_distribution: {
    POSITIVE: number;
    NEUTRAL: number;
    NEGATIVE: number;
  };
  sentiment_ratios: {
    POSITIVE: number;
    NEUTRAL: number;
    NEGATIVE: number;
  };
  top_positive_aspects: AspectFrequencyItem[];
  top_negative_aspects: AspectFrequencyItem[];
  top_neutral_aspects: AspectFrequencyItem[];
  recent_aspect_records: {
    id: string;
    prediction_id?: string | null;
    aspect: string;
    category?: string | null;
    sentiment: 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';
    confidence: number;
    start_char: number;
    end_char: number;
    evidence_text: string;
    created_at?: string | null;
  }[];
}

export interface ABSAMetadataResponse {
  model_name: string;
  version: string;
  framework: string;
  architecture: string;
  benchmark_dataset: string;
  evaluation_date: string;
  ate_metrics: {
    precision: number;
    recall: number;
    f1_score: number;
    evaluated_sentences: number;
    evaluated_aspects: number;
  };
  alsc_metrics: {
    accuracy: number;
    macro_f1: number;
    brier_score: number;
    per_class: Record<string, any>;
    confusion_matrix: string[];
  };
  baseline_comparison: {
    baseline_type: string;
    baseline_accuracy: number;
    baseline_macro_f1: number;
  };
  hardware_profile: string;
}

export interface DetectedCXCategory {
  category: string;
  score: number;
  threshold: number;
}

export interface DetectedEmotionItem {
  label: string;
  probability: number;
  threshold: number;
  cx_category: string;
}

export interface EmotionResponse {
  text: string;
  primary_emotion: string;
  confidence: number;
  detected_cx_categories: DetectedCXCategory[];
  detected_emotions: DetectedEmotionItem[];
  raw_scores: Record<string, number>;
  model_name: string;
  model_version: string;
  latency_ms: number;
  created_at: string;
}

export interface EmotionBatchResponse {
  results: EmotionResponse[];
  total_samples: number;
  latency_ms: number;
}

export interface EmotionFrequencyItem {
  label: string;
  count: number;
  ratio: number;
  cx_category: string;
}

export interface EmotionAnalyticsResponse {
  total_records: number;
  cx_distribution: Record<string, number>;
  cx_ratios: Record<string, number>;
  top_raw_emotions: EmotionFrequencyItem[];
  recent_emotion_records: {
    id: string;
    prediction_id?: string | null;
    primary_cx_category: string;
    confidence_score: number;
    detected_emotions: any[];
    created_at?: string | null;
  }[];
}

export interface EmotionMetadataResponse {
  model_name: string;
  version: string;
  framework: string;
  architecture: string;
  benchmark_dataset: string;
  license: string;
  evaluation_date: string;
  calibrated_thresholds: Record<string, number>;
  metrics: {
    micro_f1: number;
    macro_f1: number;
    hamming_loss: number;
    expected_calibration_error: number;
    mean_brier_score: number;
    per_class: Record<string, {
      precision: number;
      recall: number;
      f1: number;
      support: number;
      threshold: number;
      brier: number;
    }>;
    baseline_comparison: {
      baseline_type: string;
      micro_f1: number;
      macro_f1: number;
      hamming_loss: number;
      f1_improvement_pct: number;
    };
    sample_count: number;
  };
  cx_taxonomy_mapping: Record<string, string>;
  domain_mismatch_analysis: {
    source_domain: string;
    target_domain: string;
    mitigation_strategy: string;
  };
}

// ---------------------------------------------------------------------------
// Automatic Topic Discovery Types
// ---------------------------------------------------------------------------

export interface TopicKeywordItem {
  word: string;
  score: number;
}

export interface DiscoveredTopicItem {
  id: string;
  run_id: string;
  topic_index: number;
  auto_label: string;
  human_label?: string | null;
  display_label: string;
  keywords: TopicKeywordItem[];
  review_count: number;
  cohesion_score?: number | null;
  sentiment_breakdown?: Record<string, number> | null;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
}

export interface TopicReviewAssignmentItem {
  id: string;
  topic_id: string;
  prediction_id?: string | null;
  review_text: string;
  distance_to_centroid?: number | null;
  is_outlier: boolean;
  is_representative: boolean;
  sentiment_label?: string | null;
  confidence_score?: number | null;
  created_at: string;
}

export interface TopicDiscoveryRequest {
  source?: 'all_predictions' | 'date_range' | 'custom_texts';
  custom_texts?: string[];
  start_date?: string | null;
  end_date?: string | null;
  sentiment_filter?: string | null;
  min_cluster_size?: number;
  min_samples?: number | null;
  n_neighbors?: number;
  n_components?: number;
  random_state?: number;
}

export interface TopicDiscoveryJobStatus {
  job_id: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  progress: number;
  stage: string;
  corpus_size: number;
  num_topics: number;
  outlier_count: number;
  metrics?: Record<string, any> | null;
  error_message?: string | null;
  created_at: string;
  completed_at?: string | null;
}

export interface TopicListResponse {
  run_id: string;
  corpus_size: number;
  num_topics: number;
  outlier_count: number;
  metrics?: {
    mean_intra_cluster_cohesion?: number;
    keyword_diversity_ratio?: number;
    outlier_ratio?: number;
    cache_stats?: {
      hits: number;
      misses: number;
      total: number;
      hit_rate_pct: number;
    };
  } | null;
  topics: DiscoveredTopicItem[];
  all_runs?: {
    id: string;
    status: string;
    corpus_size: number;
    num_topics: number;
    created_at: string;
  }[];
}

export interface TopicReviewsResponse {
  topic_id: string;
  topic_name: string;
  topic_index: number;
  total_reviews: number;
  page: number;
  page_size: number;
  reviews: TopicReviewAssignmentItem[];
}

export interface MatchedTopicPair {
  topic_a_id: string;
  topic_a_name: string;
  topic_b_id: string;
  topic_b_name: string;
  similarity: number;
}

export interface CandidateEmergingTopic {
  topic_id: string;
  topic_name: string;
  topic_index: number;
  review_count: number;
  max_similarity_to_prior_run: number;
  keywords: TopicKeywordItem[];
  uncertainty_note: string;
}

export interface TopicComparisonResponse {
  run_id_a: string;
  run_id_b: string;
  matched_topics: MatchedTopicPair[];
  emerging_topics_in_b: CandidateEmergingTopic[];
  faded_topics_in_a: {
    topic_id: string;
    topic_name: string;
    topic_index: number;
    max_similarity_in_new_run: number;
  }[];
}

export interface TopicMetadataResponse {
  model_name: string;
  architecture: string;
  device: string;
  vram_usage_mb: number;
  embedding_dimension: number;
  min_reviews_required: number;
  local_weights_verified: boolean;
  benchmark_metrics: Record<string, any>;
}

// ==============================================================================
// Phase 5: Unified Review Intelligence & Cross-Feature Analytics Types
// ==============================================================================

export interface CapabilityStatusItem {
  status: 'completed' | 'unavailable' | 'pending' | 'failed';
  model_name?: string | null;
  model_version?: string | null;
  latency_ms?: number | null;
  error?: string | null;
}

export interface OverallSentimentResult {
  sentiment: 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';
  confidence: number;
  probabilities: {
    positive?: number;
    neutral?: number;
    negative?: number;
    POSITIVE?: number;
    NEUTRAL?: number;
    NEGATIVE?: number;
  };
  latency_ms?: number | null;
  explainability?: ExplainabilityToken[] | null;
}

export interface UnifiedAspectItem {
  aspect_term: string;
  aspect_category?: string | null;
  sentiment: 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';
  confidence: number;
  start_char: number;
  end_char: number;
  evidence_text?: string | null;
}

export interface UnifiedEmotionResult {
  primary_cx_category: string;
  confidence_score: number;
  detected_emotions: { emotion: string; score: number }[];
  all_probabilities: Record<string, number>;
}

export interface UnifiedTopicProjection {
  topic_id?: string | null;
  topic_index: number;
  topic_label: string;
  similarity_score?: number | null;
  topic_run_version?: string | null;
  is_outlier: boolean;
}

export interface UnifiedReviewResponse {
  review_id: string;
  text_content: string;
  overall_sentiment: OverallSentimentResult | null;
  aspects: UnifiedAspectItem[];
  emotions: UnifiedEmotionResult | null;
  topic: UnifiedTopicProjection | null;
  capability_status: {
    sentiment: CapabilityStatusItem;
    absa: CapabilityStatusItem;
    emotion: CapabilityStatusItem;
    topics: CapabilityStatusItem;
  };
  total_latency_ms: number;
  created_at: string;
}

export interface IntelligenceAnalyzeRequest {
  text: string;
  save_to_db?: boolean;
  project_topic?: boolean;
}

export interface IntelligenceBatchRequest {
  reviews: string[];
  save_to_db?: boolean;
  sync?: boolean;
}

export interface IntelligenceBatchJobStatus {
  job_id: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  progress_pct: number;
  total_records: number;
  processed_records: number;
  succeeded_records: number;
  failed_records: number;
  partial_records: number;
  error_message?: string | null;
  created_at: string;
  completed_at?: string | null;
  results?: UnifiedReviewResponse[] | null;
}

export interface AspectSentimentByTopicItem {
  topic_id: string;
  topic_name: string;
  topic_index: number;
  aspect_term: string;
  aspect_category?: string | null;
  positive_count: number;
  neutral_count: number;
  negative_count: number;
  total_count: number;
  positive_pct: number;
  negative_pct: number;
  sample_count: number;
  low_sample_warning: boolean;
}

export interface EmotionBySentimentItem {
  sentiment_label: string;
  emotion_category: string;
  count: number;
  percentage: number;
  sample_count: number;
  low_sample_warning: boolean;
}

export interface EmotionByAspectCategoryItem {
  aspect_category: string;
  emotion_category: string;
  count: number;
  percentage: number;
  sample_count: number;
  low_sample_warning: boolean;
}

export interface TopicSummaryItem {
  topic_id: string;
  topic_name: string;
  topic_index: number;
  total_reviews: number;
  sentiment_breakdown: Record<string, number>;
  dominant_emotion: string;
  sample_count: number;
  low_sample_warning: boolean;
}

export interface NegativeAspectTrendItem {
  date: string;
  aspect_term: string;
  negative_count: number;
  total_aspect_count: number;
  negative_rate: number;
  sample_count: number;
  low_sample_warning: boolean;
}

export interface IntelligenceAnalyticsResponse {
  aspect_sentiment_by_topic: AspectSentimentByTopicItem[];
  emotion_by_overall_sentiment: EmotionBySentimentItem[];
  emotion_by_aspect_category: EmotionByAspectCategoryItem[];
  topic_sentiment_and_emotion_summary: TopicSummaryItem[];
  negative_aspect_trends: NegativeAspectTrendItem[];
  methodology_note: string;
  min_sample_size_applied: number;
  total_analyzed_reviews: number;
}

export interface EvidenceItem {
  prediction_id: string;
  review_text: string;
  overall_sentiment?: string | null;
  confidence_score?: number | null;
  aspect_term?: string | null;
  aspect_sentiment?: string | null;
  emotion_category?: string | null;
  topic_name?: string | null;
  created_at: string;
}

export interface EvidenceDrilldownResponse {
  filters: Record<string, any>;
  total_matched: number;
  page: number;
  page_size: number;
  items: EvidenceItem[];
}



