import { Clock, Bell, AlertTriangle } from 'lucide-react';

const PAGE_TITLES = {
  overview:       { label: 'System Overview',       sub: 'wildfire-mlops-123 · us-central1' },
  'data-pipeline':{ label: 'Data Pipeline',          sub: 'wildfire_data_pipeline · 0 */6 * * * UTC' },
  obj1:           { label: 'OBJ-1 Ignition Classifier', sub: 'xgboost_ignition · MLflow: wildfire-ignition-v1' },
  obj2:           { label: 'OBJ-2 Fire Spread Simulator', sub: 'Rothermel (1972) + FBFM40 physics model' },
  obj3:           { label: 'OBJ-3 AI Disaster Reporter', sub: 'GeminiDisasterReporter · gemini-2.5-flash' },
  'risk-monitor': { label: 'Wildfire Risk Monitor',  sub: 'H3 64km grid · California & Texas · Jan 2025' },
  reports:        { label: 'Incident Reports',        sub: 'ICS-209 aligned · OBJ-3 Gemini LLM output' },
};

export default function Header({ activeView }) {
  const page = PAGE_TITLES[activeView] || PAGE_TITLES.overview;
  const now = new Date('2025-01-15T18:04:32Z');

  return (
    <header className="h-12 bg-surface-1 border-b border-border-subtle flex items-center justify-between px-5 flex-shrink-0">
      <div className="flex items-center gap-3">
        <div>
          <h1 className="text-text-primary text-sm font-semibold leading-tight">{page.label}</h1>
          <p className="text-text-muted text-[10px] font-mono leading-tight">{page.sub}</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Alert indicator */}
        <div className="flex items-center gap-1.5 px-2 py-1 bg-risk-critical/10 border border-risk-critical/30 rounded text-risk-critical">
          <AlertTriangle className="w-3 h-3" />
          <span className="text-[10px] font-mono font-semibold">1 CRITICAL</span>
        </div>

        {/* Clock */}
        <div className="flex items-center gap-1.5 text-text-muted">
          <Clock className="w-3 h-3" />
          <span className="text-[10px] font-mono">
            {now.toISOString().replace('T', ' ').slice(0, 19)} UTC
          </span>
        </div>

        {/* Bell */}
        <button className="relative text-text-muted hover:text-text-primary transition-colors">
          <Bell className="w-4 h-4" />
          <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-risk-high rounded-full" />
        </button>
      </div>
    </header>
  );
}
