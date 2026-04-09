import { useState } from 'react';
import {
  LayoutDashboard, GitBranch, BrainCircuit, Map, FileText, Flame,
  ChevronRight, Activity
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'overview',       label: 'Overview',         icon: LayoutDashboard, badge: null },
  { id: 'data-pipeline',  label: 'Data Pipeline',    icon: GitBranch,       badge: null },
  {
    id: 'model-pipeline', label: 'Model Pipeline',   icon: BrainCircuit,    badge: null,
    children: [
      { id: 'obj1', label: 'OBJ-1 Ignition',  icon: Activity },
      { id: 'obj2', label: 'OBJ-2 Spread Sim', icon: Flame },
      { id: 'obj3', label: 'OBJ-3 Reporter',   icon: FileText },
    ],
  },
  { id: 'risk-monitor',   label: 'Risk Monitor',     icon: Map,             badge: '1 CRIT' },
  { id: 'reports',        label: 'Incident Reports', icon: FileText,        badge: null },
];

export default function Sidebar({ activeView, onNavigate }) {
  const [modelExpanded, setModelExpanded] = useState(true);

  return (
    <aside className="w-56 min-h-screen bg-surface-1 border-r border-border-subtle flex flex-col flex-shrink-0">
      {/* Logo / project identity */}
      <div className="px-4 py-4 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <Flame className="text-risk-high w-5 h-5 flex-shrink-0" />
          <div>
            <div className="text-text-primary text-xs font-semibold leading-tight tracking-wide uppercase">Wildfire Detection</div>
            <div className="text-text-muted text-[10px] font-mono leading-tight mt-0.5">MLOps Dashboard</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2 space-y-0.5">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id || (item.children && item.children.some(c => c.id === activeView));
          const isGroupParent = Boolean(item.children);

          if (isGroupParent) {
            return (
              <div key={item.id}>
                <button
                  onClick={() => setModelExpanded(v => !v)}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded text-xs transition-colors
                    ${isActive ? 'text-text-primary bg-surface-3' : 'text-text-secondary hover:text-text-primary hover:bg-surface-2'}`}
                >
                  <Icon className="w-3.5 h-3.5 flex-shrink-0" />
                  <span className="flex-1 text-left font-medium">{item.label}</span>
                  <ChevronRight className={`w-3 h-3 transition-transform ${modelExpanded ? 'rotate-90' : ''}`} />
                </button>
                {modelExpanded && (
                  <div className="ml-5 mt-0.5 space-y-0.5 border-l border-border-subtle pl-2">
                    {item.children.map((child) => {
                      const CIcon = child.icon;
                      const childActive = activeView === child.id;
                      return (
                        <button
                          key={child.id}
                          onClick={() => onNavigate(child.id)}
                          className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs transition-colors
                            ${childActive ? 'text-text-primary bg-surface-3 font-medium' : 'text-text-secondary hover:text-text-primary hover:bg-surface-2'}`}
                        >
                          <CIcon className="w-3 h-3 flex-shrink-0" />
                          <span>{child.label}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          }

          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded text-xs transition-colors
                ${activeView === item.id ? 'text-text-primary bg-surface-3 font-medium' : 'text-text-secondary hover:text-text-primary hover:bg-surface-2'}`}
            >
              <Icon className="w-3.5 h-3.5 flex-shrink-0" />
              <span className="flex-1 text-left">{item.label}</span>
              {item.badge && (
                <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-risk-critical/20 text-risk-critical border border-risk-critical/30 leading-none">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer system status */}
      <div className="px-3 py-3 border-t border-border-subtle">
        <div className="flex items-center gap-1.5 mb-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse" />
          <span className="text-[10px] text-text-muted font-mono">DAG RUNNING</span>
        </div>
        <div className="text-[10px] text-text-muted font-mono">wildfire_data_pipeline</div>
        <div className="text-[10px] text-text-muted font-mono">Mode: QUIET  Res: 64 km</div>
      </div>
    </aside>
  );
}
