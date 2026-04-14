/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        /* Theme-switchable (CSS vars — warm-dark ↔ warm-light) */
        'surface-0':      'var(--surface-0)',
        'surface-1':      'var(--surface-1)',
        'surface-2':      'var(--surface-2)',
        'surface-3':      'var(--surface-3)',
        'border-subtle':  'var(--border-subtle)',
        'border-default': 'var(--border-default)',
        'text-primary':   'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-muted':     'var(--text-muted)',

        /* Sidebar — always dark espresso (no theme switch) */
        'sidebar':        '#1c1410',
        'sidebar-hover':  'rgba(255,255,255,0.06)',
        'sidebar-active': 'rgba(194,65,12,0.25)',
        'sidebar-text':   '#a89888',
        'sidebar-border': 'rgba(255,255,255,0.08)',

        /* Primary accent — burnt orange (fire theme) */
        'accent':         '#c2410c',
        'accent-hover':   '#9a3412',
        'accent-bg':      'rgba(194,65,12,0.07)',

        /* Secondary accents (static — used with opacity modifiers) */
        'accent-green':   '#16a34a',
        'accent-orange':  '#ea580c',
        'accent-red':     '#dc2626',
        'accent-blue':    '#3b82f6',
        'accent-purple':  '#7c3aed',

        /* Risk spectrum (matches OBJ-3 Warm Command palette) */
        'risk-critical':  '#dc2626',
        'risk-high':      '#ea580c',
        'risk-medium':    '#ca8a04',
        'risk-low':       '#16a34a',

        /* Component status */
        'status-working': '#16a34a',
        'status-partial': '#ca8a04',
        'status-broken':  '#dc2626',
        'status-planned': '#78716c',
      },
      fontFamily: {
        sans:    ['DM Sans', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Outfit', 'system-ui', 'sans-serif'],
        mono:    ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '10px',
      },
      boxShadow: {
        'card':    'var(--shadow)',
        'card-lg': 'var(--shadow-lg)',
      },
    },
  },
  plugins: [],
}
