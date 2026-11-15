import React from 'react';
import { vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CrossFeatureAnalytics } from '../components/intelligence/CrossFeatureAnalytics';
import { apiService } from '../services/api';
import {
  IntelligenceAnalyticsResponse,
  EvidenceDrilldownResponse,
} from '../types/sentiment';

vi.mock('../services/api', () => ({
  apiService: {
    getCrossFeatureAnalytics: vi.fn(),
    getEvidenceDrilldown: vi.fn(),
  },
}));

describe('CrossFeatureAnalytics Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockAnalyticsData: IntelligenceAnalyticsResponse = {
    aspect_sentiment_by_topic: [
      {
        topic_id: 'topic-001',
        topic_name: 'Battery Life',
        topic_index: 0,
        aspect_term: 'charging',
        aspect_category: 'BATTERY',
        positive_count: 8,
        neutral_count: 1,
        negative_count: 1,
        total_count: 10,
        positive_pct: 0.8,
        negative_pct: 0.1,
        sample_count: 10,
        low_sample_warning: false,
      },
      {
        topic_id: 'topic-002',
        topic_name: 'Customer Support',
        topic_index: 1,
        aspect_term: 'wait time',
        aspect_category: 'SERVICE',
        positive_count: 0,
        neutral_count: 1,
        negative_count: 3,
        total_count: 4,
        positive_pct: 0.0,
        negative_pct: 0.75,
        sample_count: 4,
        low_sample_warning: true, // Should display Low Sample badge!
      },
    ],
    emotion_by_overall_sentiment: [
      {
        sentiment_label: 'POSITIVE',
        emotion_category: 'Delight / Excitement',
        count: 12,
        percentage: 0.75,
        sample_count: 16,
        low_sample_warning: false,
      },
      {
        sentiment_label: 'NEGATIVE',
        emotion_category: 'Frustration / Anger',
        count: 8,
        percentage: 0.8,
        sample_count: 10,
        low_sample_warning: false,
      },
    ],
    emotion_by_aspect_category: [
      {
        aspect_category: 'BATTERY',
        emotion_category: 'Delight / Excitement',
        count: 6,
        percentage: 0.6,
        sample_count: 10,
        low_sample_warning: false,
      },
    ],
    topic_sentiment_and_emotion_summary: [
      {
        topic_id: 'topic-001',
        topic_name: 'Battery Life',
        topic_index: 0,
        total_reviews: 10,
        sentiment_breakdown: { POSITIVE: 0.8, NEUTRAL: 0.1, NEGATIVE: 0.1 },
        dominant_emotion: 'Delight / Excitement',
        sample_count: 10,
        low_sample_warning: false,
      },
    ],
    negative_aspect_trends: [
      {
        date: '2026-11-15',
        aspect_term: 'charging',
        negative_count: 2,
        total_aspect_count: 8,
        negative_rate: 0.25,
        sample_count: 8,
        low_sample_warning: false,
      },
    ],
    methodology_note:
      'Statistical co-occurrence groupings reflect multi-model association across review corpora rather than direct causal attribution.',
    min_sample_size_applied: 1,
    total_analyzed_reviews: 42,
  };

  const mockEvidenceData: EvidenceDrilldownResponse = {
    filters: { topic_id: 'topic-001' },
    total_matched: 1,
    page: 1,
    page_size: 15,
    items: [
      {
        prediction_id: 'pred-ev-001',
        review_text: 'Charging takes less than 30 minutes to reach 80 percent, incredible speed.',
        overall_sentiment: 'POSITIVE',
        confidence_score: 0.96,
        aspect_term: 'charging',
        aspect_sentiment: 'POSITIVE',
        emotion_category: 'Delight / Excitement',
        topic_name: 'Battery Life',
        created_at: '2026-11-15T10:00:00Z',
      },
    ],
  };

  it('renders cross-feature analytics, metrics, and methodology disclaimer', async () => {
    (apiService.getCrossFeatureAnalytics as any).mockResolvedValue(mockAnalyticsData);

    render(<CrossFeatureAnalytics />);

    await waitFor(() => {
      expect(screen.getByText('Cross-Feature Intelligence')).toBeInTheDocument();
      expect(screen.getByText('42')).toBeInTheDocument(); // Total reviews
      expect(screen.getAllByText('Battery Life').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('"charging"').length).toBeGreaterThanOrEqual(1);
    });

    // Check methodology disclosure
    expect(screen.getByText(/Statistical Methodology Disclosure:/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Statistical co-occurrence groupings reflect multi-model association/i)
    ).toBeInTheDocument();
  });

  it('displays Low Sample (n < 5) warning badge for underpowered sample groups', async () => {
    (apiService.getCrossFeatureAnalytics as any).mockResolvedValue(mockAnalyticsData);

    render(<CrossFeatureAnalytics />);

    await waitFor(() => {
      expect(screen.getAllByText(/Low Sample/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('n = 4')).toBeInTheDocument();
    });
  });

  it('updates analytics data when min_sample_size filter is changed', async () => {
    (apiService.getCrossFeatureAnalytics as any).mockResolvedValue(mockAnalyticsData);

    render(<CrossFeatureAnalytics />);

    await waitFor(() => {
      expect(apiService.getCrossFeatureAnalytics).toHaveBeenCalledWith(1);
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: '5' } });

    await waitFor(() => {
      expect(apiService.getCrossFeatureAnalytics).toHaveBeenCalledWith(5);
    });
  });

  it('opens slide-over evidence drilldown drawer and renders matched reviews', async () => {
    (apiService.getCrossFeatureAnalytics as any).mockResolvedValue(mockAnalyticsData);
    (apiService.getEvidenceDrilldown as any).mockResolvedValue(mockEvidenceData);

    render(<CrossFeatureAnalytics />);

    await waitFor(() => {
      expect(screen.getAllByText('Battery Life').length).toBeGreaterThanOrEqual(1);
    });

    // Click on row in Aspect Sentiment table to open drawer
    const row = screen.getAllByText('"charging"')[0];
    fireEvent.click(row);

    await waitFor(() => {
      expect(screen.getByText('Review Evidence Drilldown')).toBeInTheDocument();
      expect(
        screen.getByText(
          '"Charging takes less than 30 minutes to reach 80 percent, incredible speed."'
        )
      ).toBeInTheDocument();
    });

    // Press Escape to close drawer
    fireEvent.keyDown(window, { key: 'Escape' });

    await waitFor(() => {
      expect(screen.queryByText('Review Evidence Drilldown')).not.toBeInTheDocument();
    });
  });
});
