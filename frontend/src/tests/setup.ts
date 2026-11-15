// Global In-Memory LocalStorage Mock & Type Declarations for Vitest / JSDOM environment
import '@testing-library/jest-dom';

declare global {
  var describe: (name: string, fn: () => void) => void;
  var it: (name: string, fn: () => void) => void;
  var expect: any;
  var beforeEach: (fn: () => void) => void;
}

const storageMap = new Map<string, string>();

const localStorageMock = {
  getItem: (key: string) => storageMap.get(key) ?? null,
  setItem: (key: string, value: string) => storageMap.set(key, String(value)),
  removeItem: (key: string) => storageMap.delete(key),
  clear: () => storageMap.clear(),
  length: 0,
  key: (index: number) => Array.from(storageMap.keys())[index] ?? null,
};

if (typeof globalThis.localStorage === 'undefined') {
  Object.defineProperty(globalThis, 'localStorage', {
    value: localStorageMock,
    writable: true,
  });
}

// Mock ResizeObserver for Recharts ResponsiveContainer
if (typeof window !== 'undefined') {
  class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  window.ResizeObserver = ResizeObserver;
}

export {};
