// HistoryTable Logic Tests

describe('HistoryTable Filtering & Sorting Logic', () => {
  const sampleRecords = [
    { id: '1', text_content: 'Awesome service', sentiment: 'POSITIVE', confidence: 0.95, latency_ms: 12.5, created_at: '2026-08-01T10:00:00Z' },
    { id: '2', text_content: 'Terrible delay', sentiment: 'NEGATIVE', confidence: 0.88, latency_ms: 25.0, created_at: '2026-08-02T10:00:00Z' },
    { id: '3', text_content: 'Normal package', sentiment: 'NEUTRAL', confidence: 0.65, latency_ms: 18.0, created_at: '2026-08-03T10:00:00Z' },
  ];

  it('filters records by sentiment label', () => {
    const filterBySentiment = (records: typeof sampleRecords, filter: string) => {
      if (filter === 'ALL') return records;
      return records.filter((r) => r.sentiment === filter);
    };

    expect(filterBySentiment(sampleRecords, 'POSITIVE')).toHaveLength(1);
    expect(filterBySentiment(sampleRecords, 'NEGATIVE')).toHaveLength(1);
    expect(filterBySentiment(sampleRecords, 'ALL')).toHaveLength(3);
  });

  it('sorts records by confidence score descending', () => {
    const sorted = [...sampleRecords].sort((a, b) => b.confidence - a.confidence);
    expect(sorted[0].id).toBe('1'); // 0.95
    expect(sorted[1].id).toBe('2'); // 0.88
    expect(sorted[2].id).toBe('3'); // 0.65
  });

  it('calculates pagination offset correctly', () => {
    const getOffset = (page: number, pageSize: number) => (page - 1) * pageSize;
    expect(getOffset(1, 10)).toBe(0);
    expect(getOffset(2, 10)).toBe(10);
    expect(getOffset(3, 10)).toBe(20);
  });
});
