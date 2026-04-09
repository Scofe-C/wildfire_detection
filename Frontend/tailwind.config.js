/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'surface-0': '#080c14',
        'surface-1': '#0d1117',
        'surface-2': '#131b2e',
        'surface-3': '#1a2540',
        'border-subtle': '#1e2d42',
        'border-default': '#253348',
        'text-primary': '#e2e8f0',
        'text-secondary': '#8a9bbf',
        'text-muted': '#4a5978',
        'accent-green': '#10b981',
        'accent-orange': '#f59e0b',
        'accent-red': '#ef4444',
        'accent-blue': '#3b82f6',
        'accent-purple': '#8b5cf6',
        'risk-critical': '#ef4444',
        'risk-high': '#f97316',
        'risk-medium': '#f59e0b',
        'risk-low': '#10b981',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
