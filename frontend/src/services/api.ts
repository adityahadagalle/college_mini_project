import axios from 'axios';
import {
  ABSABatchResponse,
  ABSAMetadataResponse,
  ABSAResponse,
  AlertRecordItem,
  AnalyticsOverview,
  AnalyticsTrendsResponse,
  AspectAnalyticsResponse,
  BatchJobResponse,
  ChartDataset,
  DriftMetricData,
  EmotionAnalyticsResponse,
  EmotionBatchResponse,
  EmotionMetadataResponse,
  EmotionResponse,
  DiscoveredTopicItem,
  TopicComparisonResponse,
  TopicDiscoveryJobStatus,
  TopicDiscoveryRequest,
  TopicListResponse,
  TopicMetadataResponse,
  TopicReviewsResponse,
  UnifiedReviewResponse,
  IntelligenceAnalyzeRequest,
  IntelligenceBatchRequest,
  IntelligenceBatchJobStatus,
  IntelligenceAnalyticsResponse,
  EvidenceDrilldownResponse,
  ExperimentRunItem,
  ModelInfoResponse,
  ModelVersionItem,
  MlOpsSystemMetrics,
  PredictResponse,
  SystemHealthResponse,
  UserSettings,
} from '../types/sentiment';

const API_BASE_URL = '/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const apiService = {
  // Single Prediction
  predictSentiment: async (text: string): Promise<PredictResponse> => {
    const response = await apiClient.post<any>('/predict', { text });
    const raw = response.data;
    // Normalize probability keys: backend may return UPPERCASE or lowercase
    const rawProbs = raw.probabilities ?? {};
    const probabilities = {
      positive: rawProbs.positive ?? rawProbs.POSITIVE ?? 0,
      neutral:  rawProbs.neutral  ?? rawProbs.NEUTRAL  ?? 0,
      negative: rawProbs.negative ?? rawProbs.NEGATIVE ?? 0,
    };
    // Ensure they sum to 1 (guard against floating-point drift)
    const total = probabilities.positive + probabilities.neutral + probabilities.negative;
    if (total > 0) {
      probabilities.positive /= total;
      probabilities.neutral  /= total;
      probabilities.negative /= total;
    }
    return { ...raw, probabilities } as PredictResponse;
  },

  // Batch Prediction
  uploadBatchFile: async (file: File): Promise<BatchJobResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<BatchJobResponse>('/batch/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getBatchStatus: async (jobId: string): Promise<BatchJobResponse> => {
    const response = await apiClient.get<BatchJobResponse>(`/batch/status/${jobId}`);
    return response.data;
  },

  getBatchDownloadUrl: (jobId: string): string => {
    return `${API_BASE_URL}/batch/download/${jobId}`;
  },

  // History
  getHistory: async (
    search?: string,
    sentiment?: string,
    limit: number = 20,
    offset: number = 0
  ): Promise<PredictResponse[]> => {
    const params: Record<string, any> = { limit, offset };
    if (search) params.search = search;
    if (sentiment && sentiment !== 'ALL') params.sentiment_filter = sentiment;
    const response = await apiClient.get<PredictResponse[]>('/history', { params });
    return response.data;
  },

  getHistoryExportUrl: (): string => {
    return `${API_BASE_URL}/history/export`;
  },

  // Analytics & Insights
  getAnalyticsOverview: async (): Promise<AnalyticsOverview> => {
    const response = await apiClient.get<AnalyticsOverview>('/analytics/overview');
    return response.data;
  },

  getAnalyticsTrends: async (
    period: 'daily' | 'weekly' | 'monthly' | 'yearly' = 'daily'
  ): Promise<AnalyticsTrendsResponse> => {
    const response = await apiClient.get<AnalyticsTrendsResponse>(`/analytics/trends?period=${period}`);
    return response.data;
  },

  getAnalyticsCharts: async (): Promise<ChartDataset> => {
    const response = await apiClient.get<ChartDataset>('/analytics/charts');
    return response.data;
  },

  // User Settings
  getUserSettings: async (): Promise<UserSettings> => {
    const response = await apiClient.get<UserSettings>('/settings');
    return response.data;
  },

  updateUserSettings: async (settings: Partial<UserSettings>): Promise<UserSettings> => {
    const response = await apiClient.put<UserSettings>('/settings', settings);
    return response.data;
  },

  // System & Model Information
  getModelInfo: async (): Promise<ModelInfoResponse> => {
    const response = await apiClient.get<ModelInfoResponse>('/model/info');
    return response.data;
  },

  getSystemHealth: async (): Promise<SystemHealthResponse> => {
    const response = await apiClient.get<SystemHealthResponse>('/health/system');
    return response.data;
  },

  // MLOps & Observability
  getMlOpsSystemMetrics: async (): Promise<MlOpsSystemMetrics> => {
    const response = await apiClient.get<MlOpsSystemMetrics>('/monitoring/system');
    return response.data;
  },

  getRegisteredModels: async (): Promise<ModelVersionItem[]> => {
    const response = await apiClient.get<ModelVersionItem[]>('/monitoring/models');
    return response.data;
  },

  activateModelVersion: async (version: string): Promise<ModelVersionItem> => {
    const response = await apiClient.post<ModelVersionItem>(`/monitoring/models/${version}/activate`);
    return response.data;
  },

  getDriftMetrics: async (): Promise<DriftMetricData> => {
    const response = await apiClient.get<DriftMetricData>('/monitoring/drift');
    return response.data;
  },

  getExperimentRuns: async (): Promise<ExperimentRunItem[]> => {
    const response = await apiClient.get<ExperimentRunItem[]>('/monitoring/experiments');
    return response.data;
  },

  getMonitoringAlerts: async (): Promise<AlertRecordItem[]> => {
    const response = await apiClient.get<AlertRecordItem[]>('/monitoring/alerts');
    return response.data;
  },

  // Aspect-Based Sentiment Analysis (ABSA)
  analyzeAspects: async (text: string, saveToDb: boolean = true): Promise<ABSAResponse> => {
    const response = await apiClient.post<ABSAResponse>('/absa/analyze', {
      text,
      save_to_db: saveToDb,
    });
    return response.data;
  },

  analyzeAspectBatch: async (texts: string[], saveToDb: boolean = false): Promise<ABSABatchResponse> => {
    const response = await apiClient.post<ABSABatchResponse>('/absa/batch', {
      texts,
      save_to_db: saveToDb,
    });
    return response.data;
  },

  getAspectAnalytics: async (limit: number = 50): Promise<AspectAnalyticsResponse> => {
    const response = await apiClient.get<AspectAnalyticsResponse>(`/absa/analytics?limit=${limit}`);
    return response.data;
  },

  getABSAMetadata: async (): Promise<ABSAMetadataResponse> => {
    const response = await apiClient.get<ABSAMetadataResponse>('/absa/metadata');
    return response.data;
  },

  // Emotion Detection (Phase 2)
  analyzeEmotion: async (text: string, saveToDb: boolean = true): Promise<EmotionResponse> => {
    const response = await apiClient.post<EmotionResponse>('/emotion/analyze', {
      text,
      save_to_db: saveToDb,
    });
    return response.data;
  },

  analyzeEmotionBatch: async (texts: string[], saveToDb: boolean = false): Promise<EmotionBatchResponse> => {
    const response = await apiClient.post<EmotionBatchResponse>('/emotion/batch', {
      texts,
      save_to_db: saveToDb,
    });
    return response.data;
  },

  getEmotionAnalytics: async (limit: number = 50): Promise<EmotionAnalyticsResponse> => {
    const response = await apiClient.get<EmotionAnalyticsResponse>(`/emotion/analytics?limit=${limit}`);
    return response.data;
  },

  getEmotionMetadata: async (): Promise<EmotionMetadataResponse> => {
    const response = await apiClient.get<EmotionMetadataResponse>('/emotion/metadata');
    return response.data;
  },

  // Automatic Topic Discovery (Phase 3)
  discoverTopics: async (
    payload: TopicDiscoveryRequest,
    sync: boolean = true
  ): Promise<TopicListResponse> => {
    const response = await apiClient.post<TopicListResponse>(`/topics/discover?sync=${sync}`, payload);
    return response.data;
  },

  getTopicJobStatus: async (jobId: string): Promise<TopicDiscoveryJobStatus> => {
    const response = await apiClient.get<TopicDiscoveryJobStatus>(`/topics/jobs/${jobId}`);
    return response.data;
  },

  getTopics: async (runId?: string, includeArchived: boolean = false): Promise<TopicListResponse> => {
    const url = runId
      ? `/topics?run_id=${encodeURIComponent(runId)}&include_archived=${includeArchived}`
      : `/topics?include_archived=${includeArchived}`;
    const response = await apiClient.get<TopicListResponse>(url);
    return response.data;
  },

  getTopicReviews: async (
    topicId: string,
    page: number = 1,
    pageSize: number = 20
  ): Promise<TopicReviewsResponse> => {
    const response = await apiClient.get<TopicReviewsResponse>(
      `/topics/${encodeURIComponent(topicId)}/reviews?page=${page}&page_size=${pageSize}`
    );
    return response.data;
  },

  renameTopic: async (topicId: string, humanLabel: string): Promise<DiscoveredTopicItem> => {
    const response = await apiClient.patch<DiscoveredTopicItem>(
      `/topics/${encodeURIComponent(topicId)}/rename`,
      { human_label: humanLabel }
    );
    return response.data;
  },

  mergeTopics: async (
    sourceTopicId: string,
    targetTopicId: string,
    newLabel?: string
  ): Promise<DiscoveredTopicItem> => {
    const response = await apiClient.post<DiscoveredTopicItem>('/topics/merge', {
      source_topic_id: sourceTopicId,
      target_topic_id: targetTopicId,
      new_label: newLabel,
    });
    return response.data;
  },

  archiveTopic: async (topicId: string): Promise<DiscoveredTopicItem> => {
    const response = await apiClient.patch<DiscoveredTopicItem>(
      `/topics/${encodeURIComponent(topicId)}/archive`
    );
    return response.data;
  },

  compareTopicRuns: async (runIdA: string, runIdB: string): Promise<TopicComparisonResponse> => {
    const response = await apiClient.get<TopicComparisonResponse>(
      `/topics/compare?run_id_a=${encodeURIComponent(runIdA)}&run_id_b=${encodeURIComponent(runIdB)}`
    );
    return response.data;
  },

  getTopicMetadata: async (): Promise<TopicMetadataResponse> => {
    const response = await apiClient.get<TopicMetadataResponse>('/topics/metadata');
    return response.data;
  },

  // Unified Multi-Task Intelligence & Cross-Feature Analytics
  analyzeUnifiedReview: async (
    text: string,
    saveToDb: boolean = true,
    projectTopic: boolean = true
  ): Promise<UnifiedReviewResponse> => {
    const response = await apiClient.post<UnifiedReviewResponse>('/intelligence/analyze', {
      text,
      save_to_db: saveToDb,
      project_topic: projectTopic,
    });
    return response.data;
  },

  processUnifiedBatch: async (
    reviews: string[],
    saveToDb: boolean = true,
    sync: boolean = true
  ): Promise<IntelligenceBatchJobStatus> => {
    const response = await apiClient.post<IntelligenceBatchJobStatus>('/intelligence/batch', {
      reviews,
      save_to_db: saveToDb,
      sync,
    });
    return response.data;
  },

  getUnifiedBatchStatus: async (jobId: string): Promise<IntelligenceBatchJobStatus> => {
    const response = await apiClient.get<IntelligenceBatchJobStatus>(
      `/intelligence/batch/${encodeURIComponent(jobId)}`
    );
    return response.data;
  },

  getCrossFeatureAnalytics: async (
    minSampleSize: number = 1,
    startDate?: string,
    endDate?: string
  ): Promise<IntelligenceAnalyticsResponse> => {
    let url = `/intelligence/analytics?min_sample_size=${minSampleSize}`;
    if (startDate) url += `&start_date=${encodeURIComponent(startDate)}`;
    if (endDate) url += `&end_date=${encodeURIComponent(endDate)}`;
    const response = await apiClient.get<IntelligenceAnalyticsResponse>(url);
    return response.data;
  },

  getEvidenceDrilldown: async (params: {
    topic_id?: string;
    aspect_term?: string;
    emotion_category?: string;
    sentiment?: string;
    page?: number;
    page_size?: number;
  }): Promise<EvidenceDrilldownResponse> => {
    const query = new URLSearchParams();
    if (params.topic_id) query.append('topic_id', params.topic_id);
    if (params.aspect_term) query.append('aspect_term', params.aspect_term);
    if (params.emotion_category) query.append('emotion_category', params.emotion_category);
    if (params.sentiment) query.append('sentiment', params.sentiment);
    if (params.page) query.append('page', String(params.page));
    if (params.page_size) query.append('page_size', String(params.page_size));

    const response = await apiClient.get<EvidenceDrilldownResponse>(
      `/intelligence/evidence?${query.toString()}`
    );
    return response.data;
  },
};
