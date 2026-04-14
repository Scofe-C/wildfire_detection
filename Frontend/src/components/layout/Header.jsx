import { Clock, Bell, AlertTriangle, Sun, Moon } from 'lucide-react';
import { useTheme } from '../ui/ThemeProvider';
import Badge from '../ui/Badge';

const PAGE_TITLES = {
  overview:        { label: 'System Overview',              sub: 'wildfire-mlops-123 · us-central1 · Jan 2025' },
  'data-pipeline': { label: 'Data Pipeline',                sub: 'wildfire_data_pipeline · 0 */6 * * * UTC' },
  obj1:            { label: 'OBJ-1 Ignition Classifier',    sub: 'xgboost_ignition · MLflow: wildfire-ignition-v1' },
  obj2:            { label: 'OBJ-2 Fire Spread Simulator',  sub: 'Rothermel (1972) + FBFM40 physics model' },
  obj3:            { label: 'OBJ-3 AI Disaster Reporter',   sub: 'GeminiDisasterReporter · gemini-2.5-flash' },
  'fire-map':      { label: 'Fire Detection Map',           sub: 'OBJ-1 risk overlay · OBJ-2 spread · H3 64km · CA & TX' },
  'risk-monitor':  { label: 'Wildfire Risk Monitor',        sub: 'H3 64km grid · California & Texas · Jan 2025' },
  reports:         { label: 'Incident Reports',             sub: 'ICS-209 aligned · OBJ-3 Gemini LLM output' },
};

export default function Header({ activeView }) {
  const page = PAGE_TITLES[activeView] || PAGE_TITLES.overview;
  const { theme, toggle } = useTheme();
  const now = new Date('2025-01-15T18:04:32Z');

  return (
    <header className="h-14 bg-surface-1 border-b border-border-subtle flex items-center justify-between px-6 flex-shrink-0">
      <div>
        <h1 className="font-display text-text-primary text-[15px] font-semibold leading-tight">{page.label}</h1>
        <p className="text-text-muted text-[10px] font-mono leading-tight">{page.sub}</p>
      </div>

      <div className="flex items-center gap-3">
        {/* Alert badges */}
        <div className="flex items-center gap-1.5">
          <Badge color="critical">
            <AlertTriangle className="w-2.5 h-2.5" />
            2 CRITICAL
          </Badge>
          <Badge color="orange">
            <AlertTriangle className="w-2.5 h-2.5" />
            3 WARN
          </Badge>
        </div>

        {/* Clock */}
        <div className="flex items-center gap-1.5 text-text-muted">
          <Clock className="w-3 h-3" />
          <span className="text-[10px] font-mono">
            {now.toISOString().replace('T', ' ').slice(0, 19)} UTC
          </span>
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggle}
          className="w-7 h-7 flex items-center justify-center rounded-[7px] border border-border-subtle text-text-muted hover:text-text-primary hover:bg-surface-2 transition-colors"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </button>

        {/* Bell */}
        <button className="relative text-text-muted hover:text-text-primary transition-colors">
          <Bell className="w-4 h-4" />
          <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-risk-critical rounded-full" />
        </button>
      </div>
    </header>
  );
}
