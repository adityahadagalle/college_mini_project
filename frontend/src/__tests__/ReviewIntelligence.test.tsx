import React from 'react';
import { vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ReviewIntelligence } from '../components/intelligence/ReviewIntelligence';
import { apiService } from '../services/api';
import { UnifiedReviewResponse, IntelligenceBatchJobStatus } from '../types/sentiment';

// Mock apiService
vi.mock('../services/api', () => ({
  apiService: {
    analyzeUnifiedReview: vi.fn(),
    processUnifiedBatch: vi.fn(),
    getUnifiedBatchStatus: vi.fn(),
  },
}));

describe('ReviewIntelligence Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockSuccessResponse: UnifiedReviewResponse = {
    review_id: 'test-review-uuid-1',
    text_content: 'The battery life is fantastic and the screen scratches easily.',
    overall_sentiment: {
      sentiment: 'POSITIVE',
      confidence: 0.942,
      probabilities: { positive: 0.942, neutral: 0.038, negative: 0.02 },
      latency_ms: 14.2,
    },
    aspects: [
      {
        aspect_term: 'battery life',
        aspect_category: 'BATTERY',
        sentiment: 'POSITIVE',
        confidence: 0.95,
        start_char: 4,
        end_char: 16,
        evidence_text: 'The battery life is fantastic',
      },
      {
        aspect_term: 'screen',
        aspect_category: 'SCREEN',
        sentiment: 'NEGATIVE',
        confidence: 0.91,
        start_char: 38,
        end_char: 44,
        evidence_text: 'screen scratches easily',
      },
    ],
    emotions: {
      primary_cx_category: 'Delight / Excitement',
      confidence_score: 0.88,
      detected_emotions: [
        { emotion: 'admiration', score: 0.88 },
        { emotion: 'joy', score: 0.74 },
      ],
      all_probabilities: { admiration: 0.88, joy: 0.74 },
    },
    topic: {
      topic_id: 'top-battery-001',
      topic_index: 0,
      topic_label: 'Battery Performance',
      similarity_score: 0.825,
      topic_run_version: 'run-v1',
      is_outlier: false,
    },
    capability_status: {
      sentiment: { status: 'completed', model_name: 'BiLSTM', latency_ms: 14.2 },
      absa: { status: 'completed', model_name: 'DeBERTa-v3', latency_ms: 45.1 },
      emotion: { status: 'completed', model_name: 'RoBERTa', latency_ms: 32.4 },
      topics: { status: 'completed', model_name: 'all-MiniLM-L6-v2', latency_ms: 12.0 },
    },
    total_latency_ms: 103.7,
    created_at: new Date().toISOString(),
  };

  it('renders initial empty state and preset domain chips', () => {
    render(<ReviewIntelligence />);

    expect(screen.getByText('Unified Review Intelligence')).toBeInTheDocument();
    expect(screen.getByText('Review Input Workbench')).toBeInTheDocument();
    expect(screen.getByText('Ready for Intelligence Analysis')).toBeInTheDocument();
    expect(screen.getByText('Battery & Screen Hardware')).toBeInTheDocument();
  });

  it('populates textarea when clicking a preset chip', () => {
    render(<ReviewIntelligence />);

    const audioChip = screen.getByText('Audio Quality & Customer Service');
    fireEvent.click(audioChip);

    const textarea = screen.getByPlaceholderText(/Paste or enter customer review text/i);
    expect(textarea).toHaveValue(
      'Noise cancellation and sound quality are truly top-notch, though customer support was completely unhelpful when I inquired about the warranty.'
    );
  });

  it('executes unified multi-task intelligence and renders interactive evidence with aspect highlights', async () => {
    (apiService.analyzeUnifiedReview as any).mockResolvedValue(mockSuccessResponse);

    render(<ReviewIntelligence />);

    const analyzeBtn = screen.getByRole('button', { name: /Analyze Review/i });
    fireEvent.click(analyzeBtn);

    await waitFor(() => {
      expect(screen.getByText('Annotated Review Evidence')).toBeInTheDocument();
    });

    // Check that aspect buttons are rendered inside evidence
    expect(screen.getAllByText('battery life').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('screen').length).toBeGreaterThanOrEqual(1);

    // Check synchronized readout panels
    expect(screen.getByText('94.2%')).toBeInTheDocument(); // Overall confidence
    expect(screen.getByText('Delight / Excitement')).toBeInTheDocument(); // CX Emotion
    expect(screen.getByText('Battery Performance')).toBeInTheDocument(); // Topic
    expect(screen.getByText('Extracted Aspects (2)')).toBeInTheDocument(); // ABSA count
  });

  it('displays Topic Prerequisite Banner when topic discovery run is unavailable', async () => {
    const degradedResponse: UnifiedReviewResponse = {
      ...mockSuccessResponse,
      topic: null,
      capability_status: {
        ...mockSuccessResponse.capability_status,
        topics: {
          status: 'unavailable',
          model_name: 'all-MiniLM-L6-v2',
          error: 'No completed topic discovery run found.',
        },
      },
    };

    (apiService.analyzeUnifiedReview as any).mockResolvedValue(degradedResponse);
    const mockNavigate = vi.fn();

    render(<ReviewIntelligence onNavigate={mockNavigate} />);

    const analyzeBtn = screen.getByRole('button', { name: /Analyze Review/i });
    fireEvent.click(analyzeBtn);

    await waitFor(() => {
      expect(screen.getByText(/Topic Projection Prerequisite Required/i)).toBeInTheDocument();
    });

    // Test navigation button
    const navBtn = screen.getByRole('button', { name: /Explore & Discover Topics/i });
    fireEvent.click(navBtn);
    expect(mockNavigate).toHaveBeenCalledWith('topics');
  });

  it('executes batch processing and tracks batch job status metrics', async () => {
    const mockBatchJob: IntelligenceBatchJobStatus = {
      job_id: 'batch-job-1234',
      status: 'COMPLETED',
      progress_pct: 100.0,
      total_records: 2,
      processed_records: 2,
      succeeded_records: 2,
      failed_records: 0,
      partial_records: 0,
      created_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
      results: [mockSuccessResponse],
    };

    (apiService.processUnifiedBatch as any).mockResolvedValue(mockBatchJob);

    render(<ReviewIntelligence />);

    // Switch to Batch tab
    const batchTabBtn = screen.getByRole('button', { name: /Batch Processor/i });
    fireEvent.click(batchTabBtn);

    expect(screen.getByText('Unified Batch Intelligence Processor')).toBeInTheDocument();

    // Enter reviews
    const batchTextarea = screen.getByPlaceholderText(/Paste reviews here/i);
    fireEvent.change(batchTextarea, {
      target: { value: 'Great battery life and build.\nTerrible customer service.' },
    });

    const submitBtn = screen.getByRole('button', { name: /Execute Batch Intelligence/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/Job ID:/i)).toBeInTheDocument();
      expect(screen.getByText(/100\.0%/i)).toBeInTheDocument();
      expect(screen.getByText('Succeeded')).toBeInTheDocument();
    });
  });
});
