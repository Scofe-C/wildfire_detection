import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
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
import LandingPage from './components/landing/LandingPage';

// Map route paths ↔ sidebar view IDs
export const ROUTE_TO_VIEW = {
  '/':                    'landing',
  '/overview':            'overview',
  '/data-pipeline':       'data-pipeline',
  '/model-pipeline/obj1': 'obj1',
  '/model-pipeline/obj2': 'obj2',
  '/model-pipeline/obj3': 'obj3',
  '/fire-map':            'fire-map',
  '/risk-monitor':        'risk-monitor',
  '/reports':             'reports',
};

export const VIEW_TO_ROUTE = Object.fromEntries(
  Object.entries(ROUTE_TO_VIEW).map(([k, v]) => [v, k])
);

// ── Viewer mode (?mode=viewer) ─────────────────────────────────────────────
const VIEWER_TABS = [
  { id: 'fire-map', label: 'Fire Map',         path: '/fire-map' },
  { id: 'reports',  label: 'Incident Reports', path: '/reports' },
];

function ViewerLayout() {
  const navigate  = useNavigate();
  const location  = useLocation();
  const activeId  = ROUTE_TO_VIEW[location.pathname] ?? 'fire-map';

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-surface-0 bg-dot-grid">
      <div className="h-12 bg-surface-1 border-b border-border-subtle flex items-center justify-between px-4 flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-text-primary text-sm font-semibold font-display">PyroWatch</span>
          <span className="text-[9px] font-mono text-text-muted border border-border-subtle rounded px-1.5 py-0.5">VIEWER</span>
        </div>
        <div className="flex items-center gap-1">
          {VIEWER_TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => navigate(tab.path + '?mode=viewer')}
              className={`text-[11px] font-mono px-3 py-1.5 rounded-lg border transition-colors ${
                activeId === tab.id
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
        <Routes>
          <Route path="/fire-map" element={<FireMap />} />
          <Route path="/reports"  element={<IncidentReports />} />
          <Route path="*"         element={<Navigate to="/fire-map?mode=viewer" replace />} />
        </Routes>
      </main>
    </div>
  );
}

// ── Main dashboard layout ──────────────────────────────────────────────────
function DashboardLayout() {
  const navigate   = useNavigate();
  const location   = useLocation();
  const activeView = ROUTE_TO_VIEW[location.pathname] ?? 'overview';

  const onNavigate = (viewId) => navigate(VIEW_TO_ROUTE[viewId] ?? '/overview');

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface-0 bg-dot-grid">
      <Sidebar activeView={activeView} onNavigate={onNavigate} />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header activeView={activeView} />
        <main className="flex-1 overflow-hidden">
          <Routes>
            <Route path="/overview"             element={<Overview onNavigate={onNavigate} />} />
            <Route path="/data-pipeline"        element={<DataPipeline />} />
            <Route path="/model-pipeline/obj1"  element={<OBJ1Ignition />} />
            <Route path="/model-pipeline/obj2"  element={<OBJ2Spread />} />
            <Route path="/model-pipeline/obj3"  element={<OBJ3Reporter />} />
            <Route path="/fire-map"             element={<FireMap />} />
            <Route path="/risk-monitor"         element={<RiskMonitor />} />
            <Route path="/reports"              element={<IncidentReports />} />
            <Route path="*"                     element={<Navigate to="/overview" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

// ── Root ───────────────────────────────────────────────────────────────────
function AppInner() {
  const navigate = useNavigate();
  const location = useLocation();
  const isViewer = new URLSearchParams(location.search).get('mode') === 'viewer';

  if (isViewer) {
    return (
      <ThemeProvider>
        <ViewerLayout />
      </ThemeProvider>
    );
  }

  if (location.pathname === '/') {
    return (
      <ThemeProvider>
        <div className="h-screen w-screen overflow-hidden bg-surface-0">
          <LandingPage onNavigate={(viewId) => navigate(VIEW_TO_ROUTE[viewId] ?? '/overview')} />
        </div>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      <DashboardLayout />
    </ThemeProvider>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppInner />
    </BrowserRouter>
  );
}
