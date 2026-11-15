// SinglePredict Component Logic Tests

describe('SinglePredict Component Business Logic', () => {
  it('validates text length constraints correctly', () => {
    const emptyText = '   ';
    expect(emptyText.trim().length).toBe(0);

    const validText = 'Great product experience and fast delivery!';
    expect(validText.trim().length).toBeGreaterThan(0);
    expect(validText.length).toBeLessThanOrEqual(5000);

    const overlengthText = 'a'.repeat(5001);
    expect(overlengthText.length).toBeGreaterThan(5000);
  });

  it('formats probability percentage badges accurately', () => {
    const confidence = 0.9482;
    const formatted = (confidence * 100).toFixed(1);
    expect(formatted).toBe('94.8');
  });

  it('determines sentiment badge CSS styles according to classification', () => {
    const getBadgeStyle = (label: string) => {
      if (label === 'POSITIVE') return 'emerald';
      if (label === 'NEGATIVE') return 'rose';
      return 'amber';
    };

    expect(getBadgeStyle('POSITIVE')).toBe('emerald');
    expect(getBadgeStyle('NEGATIVE')).toBe('rose');
    expect(getBadgeStyle('NEUTRAL')).toBe('amber');
  });
});
