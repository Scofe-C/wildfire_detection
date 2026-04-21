import { useState, useEffect } from 'react';
import {
  CheckCircle, XCircle, AlertTriangle, Database,
  BrainCircuit, Map, ArrowRight, Terminal, RefreshCw, TrendingUp,
} from 'lucide-react';
import {
  PIPELINE_HISTORY, DATA_QUALITY_FLAGS,
  RECENT_EVENTS, COMPONENT_STATUS, PSI_MONITORING,
} from '../../data/mockPipelineData';
import { OBJ1_RUNS } from '../../data/mockModelData';
import { CALIFORNIA_CELLS, TEXAS_CELLS, getRiskTier } from '../../data/mockGridData';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { apiUrl, normalizeCell, fmt } from '../../api';

// ── helpers ───────────────────────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, sub, color = 'text-text-primary', onClick, critical }) {
  return (
    <button
      onClick={onClick}
      className={`bg-surface-2 border rounded-lg p-4 text-left hover:border-border-default transition-colors ${
        critical ? 'border-risk-critical/40 glow-critical' : 'border-border-subtle'
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <Icon className={`w-4 h-4 ${color}`} />
        {onClick && <ArrowRight className="w-3 h-3 text-text-muted" />}
      </div>
      <div className={`text-2xl font-mono font-semibold ${color} leading-none mb-1`}>{value}</div>
      <div className="text-text-secondary text-xs">{label}</div>
      {sub && <div className="text-text-muted text-[10px] font-mono mt-0.5">{sub}</div>}
    </button>
  );
}

function RiskBadge({ tier }) {
  const colors = {
    CRITICAL: 'bg-risk-critical/20 text-risk-critical border-risk-critical/50',
    HIGH:     'bg-risk-high/20 text-risk-high border-risk-high/40',
    MEDIUM:   'bg-risk-medium/20 text-risk-medium border-risk-medium/40',
    LOW:      'bg-risk-low/20 text-risk-low border-risk-low/40',
  };
  return (
    <span className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border ${colors[tier]}`}>
      {tier}
    </span>
  );
}

const STATUS_CFG = {
  working: {
    dot:   'bg-status-working',
    badge: 'bg-status-working/10 text-status-working border-status-working/30',
    border: 'border-border-subtle',
    label: 'WORKING',
  },
  partial: {
    dot:   'bg-status-partial',
    badge: 'bg-status-partial/10 text-status-partial border-status-partial/30',
    border: 'border-border-subtle',
    label: 'PARTIAL',
  },
  broken: {
    dot:   'bg-status-broken',
    badge: 'bg-status-broken/10 text-status-broken border-status-broken/40',
    border: 'border-status-broken/35 glow-critical',
    label: 'BROKEN',
  },
  planned: {
    dot:   'bg-status-planned',
    badge: 'bg-status-planned/10 text-status-planned border-status-planned/30',
    border: 'border-border-subtle',
    label: 'PLANNED',
  },
};

function ComponentCard({ label, status, note }) {
  const cfg = STATUS_CFG[status] ?? STATUS_CFG.planned;
  return (
    <div className={`bg-surface-3 border rounded p-2.5 ${cfg.border}`}>
      <div className="flex items-center gap-1.5 mb-1">
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cfg.dot}`} />
        <span className="text-text-primary text-[10px] font-semibold leading-tight flex-1 min-w-0 truncate">
          {label}
        </span>
        <span className={`text-[8px] font-mono font-bold px-1 py-0.5 rounded border flex-shrink-0 ${cfg.badge}`}>
          {cfg.label}
        </span>
      </div>
      <div className="text-text-muted text-[9px] leading-tight">{note}</div>
    </div>
  );
}

const EVENT_LEVEL_CFG = {
  error:   { bar: 'bg-risk-critical',  labelCls: 'text-risk-critical' },
  warning: { bar: 'bg-accent-orange',  labelCls: 'text-accent-orange' },
  info:    { bar: 'bg-text-muted',     labelCls: 'text-text-muted' },
};

function EventLogItem({ ts, level, component, msg }) {
  const cfg = EVENT_LEVEL_CFG[level] ?? EVENT_LEVEL_CFG.info;
  return (
    <div className="flex gap-2 py-1.5">
      <div className={`w-0.5 flex-shrink-0 self-stretch rounded-full ${cfg.bar}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 mb-0.5 flex-wrap">
          <span className={`text-[8px] font-mono font-bold ${cfg.labelCls}`}>{level.toUpperCase()}</span>
          <span className="text-[8px] font-mono text-accent-blue">{component}</span>
          <span className="text-[8px] font-mono text-text-muted ml-auto">{ts.slice(5, 16).replace('T', ' ')}</span>
        </div>
        <div className="text-[9px] text-text-secondary leading-snug">{msg}</div>
      </div>
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export default function Overview({ onNavigate }) {
  const [liveCells, setLiveCells] = useState(null);
  const [loadingCells, setLoadingCells] = useState(true);
  const [airflowRuns, setAirflowRuns] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchLive() {
      try {
        const [caRes, txRes] = await Promise.all([
          fetch(apiUrl('/api/grid-cells?region=california')),
          fetch(apiUrl('/api/grid-cells?region=texas')),
        ]);
        if (cancelled) return;
        if (caRes.ok && txRes.ok) {
          const ca = await caRes.json();
          const tx = await txRes.json();
          const cells = [...(ca.cells || []).map(c => normalizeCell({ ...c, region: 'california' })),
                         ...(tx.cells || []).map(c => normalizeCell({ ...c, region: 'texas' }))];
          setLiveCells(cells);
        }
      } catch { /* backend offline — fall back to mock */ }
      if (!cancelled) setLoadingCells(false);
    }
    fetchLive();
    const id = setInterval(fetchLive, 60_000); // refresh every 60s
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Fetch Airflow run history
  useEffect(() => {
    async function fetchAirflow() {
      try {
        const res = await fetch(apiUrl('/api/airflow/dag-runs?limit=8'));
        if (res.ok) {
          const data = await res.json();
          if (data.airflow_online && data.runs?.length) setAirflowRuns(data.runs);
        }
      } catch {}
    }
    fetchAirflow();
    const id = setInterval(fetchAirflow, 30_000);
    return () => clearInterval(id);
  }, []);

  const allCells     = liveCells || (loadingCells ? [] : [...CALIFORNIA_CELLS, ...TEXAS_CELLS]);
  const criticalCells = allCells.filter(c => getRiskTier(c.fire_risk_score) === 'CRITICAL');
  const highCells     = allCells.filter(c => getRiskTier(c.fire_risk_score) === 'HIGH');
  const prodRuns      = OBJ1_RUNS.filter(r => r.status === 'production');

  // Pipeline run history — prefer live Airflow data, fall back to mock
  const historyData = airflowRuns
    ? [...airflowRuns].reverse().map((r, i) => ({
        name: `R${i + 1}`,
        duration: r.duration_s ?? 0,
        status: r.state === 'success' ? 'success' : r.state === 'running' ? 'warning' : 'failed',
        run_id: r.run_id,
        start_date: r.start_date,
      }))
    : PIPELINE_HISTORY.slice(0, 8).map((h, i) => ({
        name: `R${i + 1}`,
        duration: h.duration_s,
        status: h.status,
      }));
  const barColor = (s) => s === 'success' ? '#10b981' : s === 'warning' ? '#f59e0b' : '#ff3333';

  // Recent events — prefer live Airflow runs, fall back to mock
  const recentEvents = airflowRuns
    ? airflowRuns.map(r => ({
        ts: r.start_date || '',
        level: r.state === 'failed' ? 'error' : r.state === 'running' ? 'warning' : 'info',
        component: 'wildfire_data_pipeline',
        msg: r.state === 'running'
          ? `DAG run in progress — started ${r.start_date ? r.start_date.slice(5, 16).replace('T', ' ') : '—'} UTC`
          : r.state === 'success'
          ? `DAG run completed · ${r.duration_s != null ? `${r.duration_s}s` : '—'} · ${r.run_id?.slice(0, 28) ?? ''}`
          : r.state === 'failed'
          ? `DAG run FAILED — ${r.run_id?.slice(0, 32) ?? ''}`
          : `DAG run queued — ${r.run_id?.slice(0, 32) ?? ''}`,
      }))
    : RECENT_EVENTS;

  const brokenCount  = COMPONENT_STATUS.filter(c => c.status === 'broken').length;
  const partialCount = COMPONENT_STATUS.filter(c => c.status === 'partial').length;
  const driftCount   = PSI_MONITORING.features.filter(f => f.status === 'drift').length;

  return (
    <div className="p-6 space-y-5 overflow-y-auto h-full">

      {/* ── 1. Operational Banner ──────────────────────────────────────────── */}
      <div className="bg-surface-2 border border-border-default rounded-lg px-4 py-2.5 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-accent-green animate-pulse flex-shrink-0" />
          <div>
            <span className="text-accent-green text-xs font-mono font-semibold">MODE: QUIET</span>
            <span className="text-text-muted text-[10px] font-mono ml-3">
              Res: 64km · Poll: 30min · Cycle: 6hr
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/10 text-risk-critical border-risk-critical/40">
            GOES-R: STUB
          </span>
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-accent-orange/10 text-accent-orange border-accent-orange/40">
            PSI DRIFT: fire_weather_index (0.31)
          </span>
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-accent-orange/10 text-accent-orange border-accent-orange/40">
            AUTO-RETRAIN: NOT IMPL.
          </span>
          <span className="text-text-muted text-[10px] font-mono">Last: 2025-01-15 18:04 UTC</span>
        </div>
      </div>

      {/* ── 2. Stat Cards ──────────────────────────────────────────────────── */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          icon={Database}
          label="Data Sources Active"
          value="4 / 5"
          sub="GOES-R: stub — not wired"
          color="text-accent-orange"
          onClick={() => onNavigate('data-pipeline')}
        />
        <StatCard
          icon={BrainCircuit}
          label="Models in Production"
          value={`${prodRuns.length} / ${OBJ1_RUNS.length}`}
          sub="LightGBM a3f1c291 failed AUC-PR gate"
          color="text-accent-blue"
          onClick={() => onNavigate('obj1')}
        />
        <StatCard
          icon={Map}
          label="Grid Cells Monitored"
          value={allCells.length}
          sub={`CA: ${CALIFORNIA_CELLS.length}  TX: ${TEXAS_CELLS.length}  ·  H3 64km`}
          color="text-text-primary"
          onClick={() => onNavigate('risk-monitor')}
        />
        <StatCard
          icon={AlertTriangle}
          label="Critical Cells"
          value={criticalCells.length}
          sub={`+${highCells.length} HIGH tier`}
          color="text-risk-critical"
          critical={criticalCells.length > 0}
          onClick={() => onNavigate('risk-monitor')}
        />
      </div>

      {/* ── 3. Component Status Matrix ─────────────────────────────────────── */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider">
            System Components
          </h2>
          <div className="flex items-center gap-2 text-[10px] font-mono">
            {brokenCount > 0 && (
              <span className="px-1.5 py-0.5 rounded border bg-status-broken/10 text-status-broken border-status-broken/30">
                {brokenCount} BROKEN
              </span>
            )}
            {partialCount > 0 && (
              <span className="px-1.5 py-0.5 rounded border bg-status-partial/10 text-status-partial border-status-partial/30">
                {partialCount} PARTIAL
              </span>
            )}
          </div>
        </div>
        <div className="grid grid-cols-4 gap-2">
          {COMPONENT_STATUS.map(c => (
            <ComponentCard key={c.id} {...c} />
          ))}
        </div>
        <div className="mt-3 pt-2 border-t border-border-subtle flex items-center gap-4 text-[9px] font-mono text-text-muted flex-wrap gap-y-1">
          {Object.entries(STATUS_CFG).map(([k, v]) => (
            <span key={k} className="flex items-center gap-1">
              <span className={`w-1.5 h-1.5 rounded-full ${v.dot}`} />
              {v.label}
            </span>
          ))}
        </div>
      </div>

      {/* ── 4. Pipeline History + Recent Events ────────────────────────────── */}
      <div className="grid grid-cols-3 gap-4">

        {/* Pipeline run history chart */}
        <div className="col-span-2 bg-surface-2 border border-border-subtle rounded-lg p-4">
          <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3 flex items-center gap-2">
            Pipeline Run History
            {airflowRuns
              ? <span className="text-[9px] font-mono text-accent-green font-normal normal-case bg-accent-green/10 border border-accent-green/25 px-1.5 py-0.5 rounded">LIVE · Airflow</span>
              : <span className="text-text-muted font-normal normal-case text-[10px] ml-1">(mock · Airflow offline)</span>
            }
          </h2>
          <ResponsiveContainer width="100%" height={100}>
            <BarChart data={historyData} margin={{ top: 2, right: 2, left: -20, bottom: 2 }}>
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#4a5978' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 9, fill: '#4a5978' }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: '#131b2e', border: '1px solid #253348', borderRadius: 4, fontSize: 10 }}
                labelStyle={{ color: '#8a9bbf' }}
                formatter={(v) => [`${v}s`, 'duration']}
              />
              <Bar dataKey="duration" radius={[2, 2, 0, 0]}>
                {historyData.map((entry, i) => (
                  <Cell key={i} fill={barColor(entry.status)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-1">
            {[['success', '#10b981'], ['warning', '#f59e0b'], ['failed', '#ff3333']].map(([s, c]) => (
              <div key={s} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm inline-block" style={{ background: c }} />
                <span className="text-[10px] text-text-muted capitalize">{s}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recent events log */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg p-4 flex flex-col">
          <div className="flex items-center gap-2 mb-2">
            <Terminal className="w-3 h-3 text-text-muted flex-shrink-0" />
            <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider flex-1">Recent Events</h2>
            {airflowRuns && <span className="text-[8px] font-mono text-accent-green">LIVE</span>}
          </div>
          <div className="flex-1 overflow-y-auto divide-y divide-border-subtle/50 max-h-[152px]">
            {recentEvents.map((e, i) => (
              <EventLogItem key={i} {...e} />
            ))}
          </div>
        </div>
      </div>

      {/* ── 5. PSI Drift + Data Quality Flags ──────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4">

        {/* PSI / Drift Monitoring */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-3 h-3 text-text-muted" />
              <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider">
                PSI / Feature Drift
              </h2>
            </div>
            <div className="flex items-center gap-2">
              {driftCount > 0 && (
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/10 text-risk-critical border-risk-critical/30">
                  {driftCount} DRIFTING
                </span>
              )}
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-accent-orange/10 text-accent-orange border-accent-orange/30">
                MANUAL
              </span>
            </div>
          </div>
          <div className="space-y-2">
            {PSI_MONITORING.features.map(f => {
              const isDrift = f.status === 'drift';
              return (
                <div key={f.feature} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-text-secondary w-40 truncate flex-shrink-0">
                    {f.feature}
                  </span>
                  <div className="flex-1 bg-surface-3 rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${isDrift ? 'bg-risk-critical' : 'bg-accent-green'}`}
                      style={{ width: `${Math.min(100, (f.psi / 0.5) * 100)}%` }}
                    />
                  </div>
                  <span className={`text-[10px] font-mono font-semibold w-8 text-right flex-shrink-0 ${
                    isDrift ? 'text-risk-critical' : 'text-accent-green'
                  }`}>
                    {f.psi.toFixed(2)}
                  </span>
                  <span className={`text-[8px] font-mono px-1 py-0.5 rounded border w-11 text-center flex-shrink-0 ${
                    isDrift
                      ? 'bg-risk-critical/10 text-risk-critical border-risk-critical/30'
                      : 'bg-accent-green/10 text-accent-green border-accent-green/30'
                  }`}>
                    {isDrift ? 'DRIFT' : 'OK'}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="mt-3 pt-2 border-t border-border-subtle text-[9px] text-text-muted font-mono">
            Ref: training_data_2024 · threshold: PSI &gt; 0.25 = drift · Auto-trigger: OFF
          </div>
        </div>

        {/* Data Quality Flags */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
          <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">
            Data Quality Flags
          </h2>
          <div className="space-y-2">
            {DATA_QUALITY_FLAGS.map(f => (
              <div key={f.flag} className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-text-muted w-6 flex-shrink-0">F{f.flag}</span>
                <div className="flex-1 bg-surface-3 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      f.flag <= 1 ? 'bg-accent-green' :
                      f.flag === 2 ? 'bg-accent-orange' :
                      f.flag === 3 ? 'bg-accent-blue' :
                      f.flag === 4 ? 'bg-accent-orange' :
                      'bg-risk-critical'
                    }`}
                    style={{ width: `${(f.count / 55) * 100}%` }}
                  />
                </div>
                <span className="text-[10px] font-mono text-text-secondary w-4 text-right flex-shrink-0">
                  {f.count}
                </span>
                <span className="text-[9px] text-text-muted w-40 truncate">{f.label}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-2 border-t border-border-subtle text-[10px] text-text-muted font-mono">
            55 cells this run · flag-1 dominant (Open-Meteo primary)
          </div>
        </div>
      </div>

      {/* ── 6. Top Risk Cells ──────────────────────────────────────────────── */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider">
            Top Risk Cells This Cycle
          </h2>
          <button
            onClick={() => onNavigate('risk-monitor')}
            className="text-[10px] text-accent-blue hover:underline flex items-center gap-1"
          >
            View all <ArrowRight className="w-3 h-3" />
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-border-subtle">
                {['grid_id', 'name', 'region', 'risk_score', 'tier', 'temp_2m', 'vpd', 'fwi', 'fuel_fbfm40', 'active_fires'].map(col => (
                  <th key={col} className="text-left text-text-muted font-mono py-1.5 pr-3 uppercase text-[9px] tracking-wider">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...allCells]
                .sort((a, b) => (b.fire_risk_score ?? 0) - (a.fire_risk_score ?? 0))
                .slice(0, 6)
                .map(cell => {
                  const tier = getRiskTier(cell.fire_risk_score);
                  const region = cell.region || (CALIFORNIA_CELLS.includes(cell) ? 'california' : 'texas');
                  const isCritical = tier === 'CRITICAL';
                  return (
                    <tr
                      key={cell.grid_id + region}
                      className={`border-b border-border-subtle/50 transition-colors ${
                        isCritical ? 'hover:bg-risk-critical/5' : 'hover:bg-surface-3/50'
                      }`}
                    >
                      <td className="py-1.5 pr-3 font-mono text-text-muted text-[10px]">
                        {cell.grid_id?.slice(0, 10)}…
                      </td>
                      <td className={`py-1.5 pr-3 font-medium ${isCritical ? 'text-risk-critical' : 'text-text-secondary'}`}>
                        {cell.name}
                      </td>
                      <td className="py-1.5 pr-3 text-text-muted font-mono capitalize">{region}</td>
                      <td className={`py-1.5 pr-3 font-mono font-semibold ${isCritical ? 'text-risk-critical' : 'text-text-primary'}`}>
                        {fmt(cell.fire_risk_score, 3)}
                      </td>
                      <td className="py-1.5 pr-3"><RiskBadge tier={tier} /></td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">{fmt(cell.temperature_2m)}°C</td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">{fmt(cell.vpd)} kPa</td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">{fmt(cell.fire_weather_index)}</td>
                      <td className="py-1.5 pr-3 font-mono text-text-muted">{cell.fuel_model_fbfm40}</td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">{cell.active_fire_count ?? 0}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── 7. Model Registry + Retrain Status ─────────────────────────────── */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider">
            Model Registry
          </h2>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded border bg-accent-orange/10 text-accent-orange border-accent-orange/30 glow-warning">
            <RefreshCw className="w-3 h-3 flex-shrink-0" />
            <span className="text-[9px] font-mono">
              Auto-Retrain: NOT IMPLEMENTED · last manual: 2025-01-10
            </span>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {OBJ1_RUNS.map(run => (
            <div
              key={run.run_id}
              className={`bg-surface-3 border rounded p-3 ${
                run.status !== 'production' ? 'border-accent-orange/30' : 'border-border-subtle'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-text-primary text-xs font-mono">{run.run_id}</span>
                <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${
                  run.status === 'production'
                    ? 'bg-accent-green/10 text-accent-green border-accent-green/30'
                    : 'bg-accent-orange/10 text-accent-orange border-accent-orange/30'
                }`}>
                  {run.status.toUpperCase()}
                </span>
              </div>
              <div className="text-text-muted text-[10px] font-mono">{run.model} · {run.region}</div>
              <div className="flex gap-3 mt-2">
                <div>
                  <div className="text-[9px] text-text-muted">AUC-PR</div>
                  <div className={`text-xs font-mono font-semibold ${
                    run.metrics.auc_pr >= 0.89 ? 'text-accent-green' : 'text-risk-critical'
                  }`}>
                    {run.metrics.auc_pr.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="text-[9px] text-text-muted">FNR</div>
                  <div className="text-xs font-mono font-semibold text-text-secondary">
                    {(run.metrics.fnr * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-[9px] text-text-muted">Threshold</div>
                  <div className="text-xs font-mono font-semibold text-text-secondary">
                    {run.metrics.threshold_tuned}
                  </div>
                </div>
              </div>
              <div className="mt-2 flex items-center gap-1">
                {run.gates.auc_pr_gate.passed
                  ? <CheckCircle className="w-3 h-3 text-accent-green" />
                  : <XCircle className="w-3 h-3 text-risk-critical" />
                }
                <span className="text-[9px] text-text-muted">AUC-PR gate</span>
                {run.gates.fnr_disparity_gate.passed
                  ? <CheckCircle className="w-3 h-3 text-accent-green ml-2" />
                  : <XCircle className="w-3 h-3 text-risk-critical ml-2" />
                }
                <span className="text-[9px] text-text-muted">Bias gate</span>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
