/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        obsidian: {
          950: '#09090B',
          900: '#0F1115',
          850: '#12141A',
          800: '#161920',
          700: '#222630',
        },
        background: '#09090B',
        panel: '#0F1115',
        card: '#161920',
        border: '#27272A',
        primary: '#FAFAFA',
        secondary: '#A1A1AA',
        positive: '#10B981',
        negative: '#F43F5E',
        warning: '#F59E0B',
      },
      boxShadow: {
        'obsidian-sm': '0 1px 2px 0 rgba(0, 0, 0, 0.4)',
        'obsidian-md': '0 4px 16px -2px rgba(0, 0, 0, 0.5)',
        'obsidian-lg': '0 10px 30px -4px rgba(0, 0, 0, 0.65)',
      },
      transitionTimingFunction: {
        'obsidian': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
    },
  },
  plugins: [],
}