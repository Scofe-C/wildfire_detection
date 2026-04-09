import { CheckCircle, XCircle, AlertTriangle, Activity, Database, BrainCircuit, Map, Clock, ArrowRight } from 'lucide-react';
import { PIPELINE_META, PIPELINE_HISTORY, DATA_QUALITY_FLAGS } from '../../data/mockPipelineData';
import { OBJ1_RUNS, OBJ3_STATE } from '../../data/mockModelData';
import { CALIFORNIA_CELLS, TEXAS_CELLS, getRiskTier } from '../../data/mockGridData';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function StatusDot({ status }) {
  const map = {
    success: 'bg-accent-green',
    warning: 'bg-accent-orange',
    failed:  'bg-risk-critical',
    running: 'bg-accent-blue animate-pulse',
    cached:  'bg-text-muted',
    stub:    'bg-text-muted',
  };
  return <span className={`inline-block w-2 h-2 rounded-full ${map[status] ?? 'bg-text-muted'}`} />;
}

function StatCard({ icon: Icon, label, value, sub, color = 'text-text-primary', onClick }) {
  return (
    <button
      onClick={onClick}
      className="bg-surface-2 border border-border-subtle rounded-lg p-4 text-left hover:border-border-default transition-colors"
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
    CRITICAL: 'bg-risk-critical/20 text-risk-critical border-risk-critical/40',
    HIGH:     'bg-risk-high/20 text-risk-high border-risk-high/40',
    MEDIUM:   'bg-risk-medium/20 text-risk-medium border-risk-medium/40',
    LOW:      'bg-risk-low/20 text-risk-low border-risk-low/40',
  };
  return (
    <span className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border ${colors[tier]}`}>{tier}</span>
  );
}

export default function Overview({ onNavigate }) {
  const allCells = [...CALIFORNIA_CELLS, ...TEXAS_CELLS];
  const criticalCells = allCells.filter(c => getRiskTier(c.fire_risk_score) === 'CRITICAL');
  const highCells     = allCells.filter(c => getRiskTier(c.fire_risk_score) === 'HIGH');

  const prodRuns = OBJ1_RUNS.filter(r => r.status === 'production');
  const recentHistory = PIPELINE_HISTORY.slice(0, 8);

  const historyBarData = recentHistory.map((h, i) => ({
    name: `R${i + 1}`,
    duration: h.duration_s,
    status: h.status,
  }));

  const barColor = (s) => {
    if (s === 'success') return '#10b981';
    if (s === 'warning') return '#f59e0b';
    return '#ef4444';
  };

  const qualityData = DATA_QUALITY_FLAGS.filter(f => f.count > 0).map(f => ({
    name: `Flag ${f.flag}`,
    label: f.label,
    count: f.count,
  }));

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      {/* Operational mode banner */}
      <div className="bg-surface-2 border border-accent-green/30 rounded-lg px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-accent-green animate-pulse" />
          <div>
            <span className="text-accent-green text-xs font-mono font-semibold">OPERATIONAL MODE: QUIET</span>
            <span className="text-text-muted text-xs font-mono ml-3">Resolution: 64 km  |  Poll: 30 min  |  Cycle: 6 hr</span>
          </div>
        </div>
        <div className="text-text-muted text-[10px] font-mono">
          Last run: 2025-01-15 18:04 UTC &nbsp;|&nbsp; Next: 2025-01-16 00:00 UTC
        </div>
      </div>

      {/* Top stat cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          icon={Database}
          label="Data Pipeline"
          value="6 / 6"
          sub="stages healthy"
          color="text-accent-green"
          onClick={() => onNavigate('data-pipeline')}
        />
        <StatCard
          icon={BrainCircuit}
          label="Models in Production"
          value={`${prodRuns.length} / ${OBJ1_RUNS.length}`}
          sub={`AUC-PR best: ${Math.max(...prodRuns.map(r => r.metrics.auc_pr)).toFixed(4)}`}
          color="text-accent-blue"
          onClick={() => onNavigate('obj1')}
        />
        <StatCard
          icon={Map}
          label="Grid Cells Monitored"
          value={allCells.length}
          sub={`CA: ${CALIFORNIA_CELLS.length}  TX: ${TEXAS_CELLS.length}`}
          color="text-text-primary"
          onClick={() => onNavigate('risk-monitor')}
        />
        <StatCard
          icon={AlertTriangle}
          label="Critical Cells"
          value={criticalCells.length}
          sub={`+${highCells.length} HIGH tier cells`}
          color={criticalCells.length > 0 ? 'text-risk-critical' : 'text-accent-green'}
          onClick={() => onNavigate('risk-monitor')}
        />
      </div>

      {/* Middle section: pipeline runs + top alerts */}
      <div className="grid grid-cols-3 gap-4">

        {/* Pipeline run history */}
        <div className="col-span-2 bg-surface-2 border border-border-subtle rounded-lg p-4">
          <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">
            Pipeline Run History  <span className="text-text-muted font-normal normal-case">(last 8 × 6-hr runs)</span>
          </h2>
          <ResponsiveContainer width="100%" height={110}>
            <BarChart data={historyBarData} margin={{ top: 2, right: 2, left: -20, bottom: 2 }}>
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#4a5978' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 9, fill: '#4a5978' }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: '#131b2e', border: '1px solid #253348', borderRadius: 4, fontSize: 10 }}
                labelStyle={{ color: '#8a9bbf' }}
                formatter={(v, n, p) => [`${v}s`, 'duration']}
              />
              <Bar dataKey="duration" radius={[2, 2, 0, 0]}>
                {historyBarData.map((entry, index) => (
                  <Cell key={index} fill={barColor(entry.status)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-2">
            {[['success', '#10b981'], ['warning', '#f59e0b'], ['failed', '#ef4444']].map(([s, c]) => (
              <div key={s} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm inline-block" style={{ background: c }} />
                <span className="text-[10px] text-text-muted capitalize">{s}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Data quality flag distribution */}
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
                    className={`h-full rounded-full ${f.flag === 0 || f.flag === 1 ? 'bg-accent-green' : f.flag === 2 || f.flag === 4 ? 'bg-accent-orange' : f.flag >= 3 ? 'bg-risk-critical' : 'bg-text-muted'}`}
                    style={{ width: `${(f.count / 55) * 100}%` }}
                  />
                </div>
                <span className="text-[10px] font-mono text-text-secondary w-4 text-right">{f.count}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-border-subtle text-[10px] text-text-muted font-mono">
            55 cells total this run
          </div>
        </div>
      </div>

      {/* Alert cells table */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider">
            Top Risk Cells This Cycle
          </h2>
          <button onClick={() => onNavigate('risk-monitor')} className="text-[10px] text-accent-blue hover:underline flex items-center gap-1">
            View all <ArrowRight className="w-3 h-3" />
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-border-subtle">
                {['grid_id', 'name', 'region', 'risk_score', 'tier', 'temp_2m', 'vpd', 'fwi', 'fuel_fbfm40', 'active_fires'].map(col => (
                  <th key={col} className="text-left text-text-muted font-mono py-1.5 pr-3 uppercase text-[9px] tracking-wider">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {allCells
                .sort((a, b) => b.fire_risk_score - a.fire_risk_score)
                .slice(0, 6)
                .map(cell => {
                  const tier = getRiskTier(cell.fire_risk_score);
                  const region = CALIFORNIA_CELLS.includes(cell) ? 'california' : 'texas';
                  return (
                    <tr key={cell.grid_id} className="border-b border-border-subtle/50 hover:bg-surface-3/50 transition-colors">
                      <td className="py-1.5 pr-3 font-mono text-text-muted text-[10px]">{cell.grid_id.slice(0, 10)}…</td>
                      <td className="py-1.5 pr-3 text-text-secondary">{cell.name}</td>
                      <td className="py-1.5 pr-3 text-text-muted font-mono capitalize">{region}</td>
                      <td className="py-1.5 pr-3 font-mono text-text-primary">{cell.fire_risk_score.toFixed(3)}</td>
                      <td className="py-1.5 pr-3"><RiskBadge tier={tier} /></td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">{cell.temperature_2m}°C</td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">{cell.vpd} kPa</td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">{cell.fire_weather_index}</td>
                      <td className="py-1.5 pr-3 font-mono text-text-muted">{cell.fuel_model_fbfm40}</td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">{cell.active_fire_count}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model health strip */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <h2 className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">Model Registry Status</h2>
        <div className="grid grid-cols-3 gap-3">
          {OBJ1_RUNS.map(run => (
            <div key={run.run_id} className="bg-surface-3 border border-border-subtle rounded p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-text-primary text-xs font-mono">{run.run_id}</span>
                <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${run.status === 'production' ? 'bg-accent-green/10 text-accent-green border-accent-green/30' : 'bg-accent-orange/10 text-accent-orange border-accent-orange/30'}`}>
                  {run.status.toUpperCase()}
                </span>
              </div>
              <div className="text-text-muted text-[10px] font-mono">{run.model} · {run.region}</div>
              <div className="flex gap-3 mt-2">
                <div>
                  <div className="text-[9px] text-text-muted">AUC-PR</div>
                  <div className={`text-xs font-mono font-semibold ${run.metrics.auc_pr >= 0.89 ? 'text-accent-green' : 'text-risk-critical'}`}>{run.metrics.auc_pr.toFixed(4)}</div>
                </div>
                <div>
                  <div className="text-[9px] text-text-muted">FNR</div>
                  <div className="text-xs font-mono font-semibold text-text-secondary">{(run.metrics.fnr * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-[9px] text-text-muted">Threshold</div>
                  <div className="text-xs font-mono font-semibold text-text-secondary">{run.metrics.threshold_tuned}</div>
                </div>
              </div>
              <div className="mt-2 flex items-center gap-1">
                {run.gates.auc_pr_gate.passed ? <CheckCircle className="w-3 h-3 text-accent-green" /> : <XCircle className="w-3 h-3 text-risk-critical" />}
                <span className="text-[9px] text-text-muted">AUC-PR gate</span>
                {run.gates.fnr_disparity_gate.passed ? <CheckCircle className="w-3 h-3 text-accent-green ml-2" /> : <XCircle className="w-3 h-3 text-risk-critical ml-2" />}
                <span className="text-[9px] text-text-muted">Bias gate</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
