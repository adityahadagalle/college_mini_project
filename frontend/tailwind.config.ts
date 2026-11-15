import type { Config } from 'tailwindcss';

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#0F172A',
        card: '#1E293B',
        border: '#334155',
        primary: '#3B82F6',
        secondary: '#6366F1',
        positive: '#10B981',
        negative: '#EF4444',
        warning: '#F59E0B',
      }
    },
  },
  plugins: [],
} satisfies Config;
