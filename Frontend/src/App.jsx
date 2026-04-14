import { useState } from 'react';
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

export default function App() {
  const [activeView, setActiveView] = useState('overview');

  const ViewComponent = VIEWS[activeView] ?? Overview;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface-0">
      <Sidebar activeView={activeView} onNavigate={setActiveView} />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header activeView={activeView} />
        <main className="flex-1 overflow-hidden">
          <ViewComponent onNavigate={setActiveView} />
        </main>
      </div>
    </div>
  );
}
