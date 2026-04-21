import { useState, useEffect } from 'react';
import {
  LayoutDashboard, GitBranch, BrainCircuit, Map, FileText, Flame,
  ChevronRight, Activity, Layers, ChevronLeft,
} from 'lucide-react';
import StatusDot from '../ui/StatusDot';
import { apiUrl } from '../../api';

const NAV_ITEMS = [
  { id: 'overview',       label: 'Overview',           icon: LayoutDashboard },
  { id: 'data-pipeline',  label: 'Data Pipeline',      icon: GitBranch },
  {
    id: 'model-pipeline', label: 'Model Pipeline',     icon: BrainCircuit,
    children: [
      { id: 'obj1', label: 'OBJ-1 Ignition',   icon: Activity },
      { id: 'obj2', label: 'OBJ-2 Spread Sim',  icon: Flame },
      { id: 'obj3', label: 'OBJ-3 Reporter',    icon: FileText },
    ],
  },
  { id: 'fire-map',       label: 'Fire Detection Map', icon: Layers },
  { id: 'risk-monitor',   label: 'Risk Monitor',       icon: Map },
  { id: 'reports',        label: 'Incident Reports',   icon: FileText },
];

// Derive display label + StatusDot status from Airflow response
function dagDisplayState(dag) {
  if (!dag) return { label: 'DAG: LOADING', dot: 'planned', badge: null };
  if (!dag.airflow_online) return { label: 'DAG: OFFLINE', dot: 'broken', badge: null };
  if (dag.is_paused) return { label: 'DAG: PAUSED', dot: 'partial', badge: null };
  const s = dag.last_run_state;
  if (s === 'running')  return { label: 'DAG: RUNNING', dot: 'working', badge: null };
  if (s === 'failed')   return { label: 'DAG: FAILED',  dot: 'broken',  badge: 'critical' };
  if (s === 'success')  return { label: 'DAG: IDLE',    dot: 'working', badge: null };
  return { label: 'DAG: UNKNOWN', dot: 'partial', badge: null };
}

export default function Sidebar({ activeView, onNavigate }) {
  const [modelExpanded, setModelExpanded] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const [dagStatus, setDagStatus] = useState(null);

  useEffect(() => {
    async function fetchDagStatus() {
      try {
        const res = await fetch(apiUrl('/api/airflow/dag-status'));
        if (res.ok) setDagStatus(await res.json());
      } catch {}
    }
    fetchDagStatus();
    const id = setInterval(fetchDagStatus, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <aside
      className="min-h-screen bg-sidebar flex flex-col flex-shrink-0 transition-[width] duration-[250ms]"
      style={{ width: collapsed ? 58 : 240 }}
    >
      {/* Logo + collapse toggle */}
      <div className="px-4 py-4 border-b border-sidebar-border">
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setCollapsed(v => !v)}
            className="flex-shrink-0 hover:opacity-80 transition-opacity"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <img
              src="/gemini-svg.svg"
              alt="PyroWatch"
              className="w-[42px] h-[42px] rounded-[10px] object-cover"
            />
          </button>
          {!collapsed && (
            <div className="flex-1 overflow-hidden">
              <div className="font-display text-[20px] font-bold text-white leading-tight whitespace-nowrap">
                PyroWatch
              </div>
              <div className="text-[10px] font-mono text-sidebar-text uppercase tracking-wider leading-tight">
                Intelligence Platform
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto overflow-x-hidden">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id || (item.children && item.children.some(c => c.id === activeView));
          const isGroupParent = Boolean(item.children);

          if (isGroupParent) {
            return (
              <div key={item.id}>
                <button
                  onClick={() => setModelExpanded(v => !v)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-[7px] text-[13px] transition-colors
                    ${isActive ? 'text-white bg-sidebar-active' : 'text-sidebar-text hover:text-white hover:bg-sidebar-hover'}`}
                >
                  <Icon className="w-4 h-4 flex-shrink-0" />
                  {!collapsed && (
                    <>
                      <span className="flex-1 text-left font-medium">{item.label}</span>
                      <ChevronRight className={`w-3 h-3 transition-transform ${modelExpanded ? 'rotate-90' : ''}`} />
                    </>
                  )}
                </button>
                {modelExpanded && !collapsed && (
                  <div className="ml-6 mt-0.5 space-y-0.5 border-l border-sidebar-border pl-2">
                    {item.children.map((child) => {
                      const CIcon = child.icon;
                      const childActive = activeView === child.id;
                      return (
                        <button
                          key={child.id}
                          onClick={() => onNavigate(child.id)}
                          className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-[5px] text-[12px] transition-colors
                            ${childActive ? 'text-white bg-sidebar-active font-medium' : 'text-sidebar-text hover:text-white hover:bg-sidebar-hover'}`}
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
              className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-[7px] text-[13px] transition-colors
                ${activeView === item.id ? 'text-white bg-sidebar-active font-medium' : 'text-sidebar-text hover:text-white hover:bg-sidebar-hover'}`}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="flex-1 text-left">{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-3 py-3 border-t border-sidebar-border">
        <div className="flex items-center justify-between">
          {!collapsed && (() => {
            const { label, dot } = dagDisplayState(dagStatus);
            return (
              <div className="flex items-center gap-2">
                <StatusDot status={dot} />
                <span className="text-[10px] font-mono text-sidebar-text">{label}</span>
              </div>
            );
          })()}
          <button
            onClick={() => setCollapsed(v => !v)}
            className="w-6 h-6 flex items-center justify-center rounded text-sidebar-text hover:text-white hover:bg-sidebar-hover transition-colors"
          >
            <ChevronLeft className={`w-3.5 h-3.5 transition-transform ${collapsed ? 'rotate-180' : ''}`} />
          </button>
        </div>
        {!collapsed && dagStatus?.airflow_online && dagStatus?.last_run_start && (
          <div className="mt-1.5 text-[9px] font-mono text-sidebar-text/60 truncate">
            {dagStatus.last_run_start.slice(0, 16).replace('T', ' ')} UTC
          </div>
        )}
        {!collapsed && dagStatus && !dagStatus.airflow_online && (
          <div className="mt-1.5 text-[9px] font-mono text-sidebar-text/50">
            Airflow unreachable
          </div>
        )}
      </div>
    </aside>
  );
}
