import { useState } from 'react';
import ThemeProvider from './components/ui/ThemeProvider';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import Overview from './components/overview/Overview';
import DataPipeline from './components/data-pipeline/DataPipeline';
import OBJ1Ignition from './components/model-pipeline/OBJ1Ignition';
import OBJ2Spread from './components/model-pipeline/OBJ2Spread';
import OBJ3Reporter from './components/model-pipeline/OBJ3Reporter';
import RiskMonitor from './components/risk-monitor/RiskMonitor';
import IncidentReports from './components/reports/IncidentReports';
import FireMap from './components/fire-map/FireMap';

const VIEWS = {
  'overview':       Overview,
  'data-pipeline':  DataPipeline,
  'obj1':           OBJ1Ignition,
  'obj2':           OBJ2Spread,
  'obj3':           OBJ3Reporter,
  'risk-monitor':   RiskMonitor,
  'reports':        IncidentReports,
  'fire-map':       FireMap,
};

// Viewer mode: only these tabs are accessible via ?mode=viewer
const VIEWER_TABS = [
  { id: 'fire-map', label: 'Fire Map' },
  { id: 'reports',  label: 'Incident Reports' },
];

function ViewerApp() {
  const [activeView, setActiveView] = useState('fire-map');
  const ViewComponent = VIEWS[activeView] ?? FireMap;

  return (
    <ThemeProvider>
      <div className="flex flex-col h-screen w-screen overflow-hidden bg-surface-0 bg-dot-grid">
        {/* Minimal viewer top bar */}
        <div className="h-12 bg-surface-1 border-b border-border-subtle flex items-center justify-between px-4 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-text-primary text-sm font-semibold font-display">PyroWatch</span>
            <span className="text-[9px] font-mono text-text-muted border border-border-subtle rounded px-1.5 py-0.5">VIEWER</span>
          </div>
          <div className="flex items-center gap-1">
            {VIEWER_TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveView(tab.id)}
                className={`text-[11px] font-mono px-3 py-1.5 rounded-lg border transition-colors ${
                  activeView === tab.id
                    ? 'bg-accent-blue/15 border-accent-blue/40 text-accent-blue font-semibold'
                    : 'border-border-subtle text-text-muted hover:text-text-primary hover:bg-surface-2'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
        <main className="flex-1 overflow-hidden">
          <ViewComponent />
        </main>
      </div>
    </ThemeProvider>
  );
}

export default function App() {
  // Detect ?mode=viewer in URL
  const isViewer = new URLSearchParams(window.location.search).get('mode') === 'viewer';
  if (isViewer) return <ViewerApp />;

  const [activeView, setActiveView] = useState('overview');
  const ViewComponent = VIEWS[activeView] ?? Overview;

  return (
    <ThemeProvider>
      <div className="flex h-screen w-screen overflow-hidden bg-surface-0 bg-dot-grid">
        <Sidebar activeView={activeView} onNavigate={setActiveView} />
        <div className="flex flex-col flex-1 overflow-hidden">
          <Header activeView={activeView} />
          <main className="flex-1 overflow-hidden">
            <ViewComponent onNavigate={setActiveView} />
          </main>
        </div>
      </div>
    </ThemeProvider>
  );
}
