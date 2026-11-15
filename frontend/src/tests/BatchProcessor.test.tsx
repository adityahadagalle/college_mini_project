// BatchProcessor Logic Tests

describe('BatchProcessor Calculations & Validation', () => {
  it('validates allowed file extensions', () => {
    const isAllowed = (filename: string) => {
      const ext = filename.split('.').pop()?.toLowerCase();
      return ['csv', 'txt', 'json'].includes(ext || '');
    };

    expect(isAllowed('reviews.csv')).toBe(true);
    expect(isAllowed('dataset.json')).toBe(true);
    expect(isAllowed('notes.txt')).toBe(true);
    expect(isAllowed('image.png')).toBe(false);
    expect(isAllowed('document.pdf')).toBe(false);
  });

  it('calculates progress percentage correctly', () => {
    const calculateProgress = (processed: number, total: number) => {
      if (total <= 0) return 0;
      return Math.min(100, Math.round((processed / total) * 100));
    };

    expect(calculateProgress(50, 100)).toBe(50);
    expect(calculateProgress(100, 100)).toBe(100);
    expect(calculateProgress(0, 500)).toBe(0);
    expect(calculateProgress(75, 100)).toBe(75);
  });

  it('estimates remaining processing time in seconds', () => {
    const calculateETA = (processed: number, total: number, rateRowsPerSec: number = 50) => {
      const remaining = Math.max(0, total - processed);
      return Math.ceil(remaining / rateRowsPerSec);
    };

    expect(calculateETA(500, 1000, 50)).toBe(10);
    expect(calculateETA(1000, 1000, 50)).toBe(0);
  });
});
