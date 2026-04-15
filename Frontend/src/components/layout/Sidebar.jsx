import { useState } from 'react';
import {
  LayoutDashboard, GitBranch, BrainCircuit, Map, FileText, Flame,
  ChevronRight, Activity, Layers, ChevronLeft, Play, RotateCcw,
} from 'lucide-react';
import Badge from '../ui/Badge';
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

export default function Sidebar({ activeView, onNavigate }) {
  const [modelExpanded, setModelExpanded] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className="min-h-screen bg-sidebar flex flex-col flex-shrink-0 transition-[width] duration-[250ms]"
      style={{ width: collapsed ? 58 : 240 }}
    >
      {/* Logo */}
      <div className="px-4 py-4 border-b border-sidebar-border">
        <div className="flex items-center gap-2.5">
          <img
            src="/gemini-svg.svg"
            alt="PyroWatch"
            className="w-[42px] h-[42px] rounded-[10px] flex-shrink-0 object-cover"
          />
          {!collapsed && (
            <div className="overflow-hidden">
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

      {/* Pipeline quick-actions */}
      {!collapsed && (
        <div className="px-3 py-2 border-t border-sidebar-border space-y-1">
          <div className="text-[9px] font-mono text-sidebar-text uppercase tracking-widest mb-1">Pipeline</div>
          <button
            onClick={() => window.open('http://localhost:8080', '_blank')}
            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-[5px] text-[11px] font-mono text-sidebar-text hover:text-white hover:bg-sidebar-hover transition-colors"
          >
            <Play className="w-3 h-3" />
            <span>Open Airflow UI</span>
          </button>
          <button
            onClick={async () => {
              try { await fetch(apiUrl('/api/pipeline/trigger'), { method: 'POST' }); } catch {}
            }}
            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-[5px] text-[11px] font-mono text-sidebar-text hover:text-white hover:bg-sidebar-hover transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Trigger DAG Run</span>
          </button>
        </div>
      )}

      {/* Footer */}
      <div className="px-3 py-3 border-t border-sidebar-border">
        <div className="flex items-center justify-between">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <StatusDot status="working" />
              <span className="text-[10px] font-mono text-sidebar-text">DAG: RUNNING</span>
            </div>
          )}
          <button
            onClick={() => setCollapsed(v => !v)}
            className="w-6 h-6 flex items-center justify-center rounded text-sidebar-text hover:text-white hover:bg-sidebar-hover transition-colors"
          >
            <ChevronLeft className={`w-3.5 h-3.5 transition-transform ${collapsed ? 'rotate-180' : ''}`} />
          </button>
        </div>
        {!collapsed && (
          <div className="mt-1.5 flex gap-1.5">
            <Badge color="critical">1 BROKEN</Badge>
            <Badge color="medium">3 PARTIAL</Badge>
          </div>
        )}
      </div>
    </aside>
  );
}
