import { CheckCircle, XCircle } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, ReferenceLine, Legend
} from 'recharts';
import { OBJ1_RUNS, SHAP_IMPORTANCE, BIAS_ANALYSIS, PR_CURVE_CA } from '../../data/mockModelData';

function KV({ label, value, mono = true, highlight }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-text-muted text-[10px] w-40 flex-shrink-0">{label}</span>
      <span className={`text-[10px] ${mono ? 'font-mono' : ''} ${highlight ? 'text-accent-green font-semibold' : 'text-text-secondary'}`}>{value}</span>
    </div>
  );
}

function GateRow({ label, threshold, value, passed }) {
  return (
    <div className="flex items-center gap-2 py-1.5 border-b border-border-subtle/50">
      {passed
        ? <CheckCircle className="w-3.5 h-3.5 text-accent-green flex-shrink-0" />
        : <XCircle    className="w-3.5 h-3.5 text-risk-critical flex-shrink-0" />
      }
      <span className="text-[10px] text-text-secondary flex-1">{label}</span>
      <span className="text-[10px] font-mono text-text-muted">threshold: {threshold}</span>
      <span className={`text-[10px] font-mono font-semibold ${passed ? 'text-accent-green' : 'text-risk-critical'}`}>{value}</span>
    </div>
  );
}

function ConfusionMatrix({ cm }) {
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  return (
    <div>
      <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Confusion Matrix</div>
      <div className="grid grid-cols-2 gap-1 text-center">
        {[
          { label: 'TN', value: cm.tn, color: 'text-accent-green', bg: 'bg-accent-green/10' },
          { label: 'FP', value: cm.fp, color: 'text-risk-high',   bg: 'bg-risk-high/10' },
          { label: 'FN', value: cm.fn, color: 'text-risk-critical', bg: 'bg-risk-critical/10' },
          { label: 'TP', value: cm.tp, color: 'text-accent-blue', bg: 'bg-accent-blue/10' },
        ].map(c => (
          <div key={c.label} className={`${c.bg} border border-border-subtle rounded p-2`}>
            <div className={`text-lg font-mono font-bold ${c.color}`}>{c.value.toLocaleString()}</div>
            <div className="text-[9px] text-text-muted font-mono">{c.label}</div>
            <div className="text-[9px] text-text-muted">{((c.value / total) * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-1 mt-1 text-[9px] text-text-muted text-center">
        <div>Predicted NEG</div>
        <div>Predicted POS</div>
      </div>
    </div>
  );
}

export default function OBJ1Ignition() {
  const prodRuns  = OBJ1_RUNS.filter(r => r.status === 'production');
  const allRuns   = OBJ1_RUNS;

  const shapData = SHAP_IMPORTANCE.map(d => ({ ...d, name: d.feature }));

  return (
    <div className="p-6 overflow-y-auto h-full space-y-5">

      {/* Header strip */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Objective</div>
            <div className="text-text-primary text-xs font-semibold">OBJ-1: Fire Ignition Classifier</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Models</div>
            <div className="text-text-secondary text-xs font-mono">xgboost_ignition · lightgbm_ignition</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">MLflow experiment</div>
            <div className="text-text-secondary text-xs font-mono">wildfire-ignition-v1</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">AUC-PR gate</div>
            <div className="text-accent-green text-xs font-mono font-semibold">≥ 0.89</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Target recall</div>
            <div className="text-accent-green text-xs font-mono font-semibold">≥ 90%</div>
          </div>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-1 bg-accent-green/10 border border-accent-green/30 rounded">
          <CheckCircle className="w-3 h-3 text-accent-green" />
          <span className="text-[10px] font-mono text-accent-green font-semibold">ALL GATES PASS</span>
        </div>
      </div>

      {/* Run cards */}
      <div className="grid grid-cols-3 gap-4">
        {allRuns.map(run => (
          <div key={run.run_id} className={`bg-surface-2 border rounded-lg p-4 ${run.status === 'production' ? 'border-accent-green/40' : 'border-border-subtle'}`}>
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-text-primary text-xs font-mono font-semibold">{run.run_id}</div>
                <div className="text-text-muted text-[10px] font-mono">{run.model}</div>
              </div>
              <span className={`text-[9px] font-mono font-bold px-2 py-1 rounded border ${
                run.status === 'production'
                  ? 'bg-accent-green/10 border-accent-green/30 text-accent-green'
                  : 'bg-accent-orange/10 border-accent-orange/30 text-accent-orange'
              }`}>{run.status.toUpperCase()}</span>
            </div>

            <div className="text-text-muted text-[9px] font-mono mb-1">{run.region} · {run.test_period}</div>

            {/* Metric grid */}
            <div className="grid grid-cols-3 gap-2 my-3">
              {[
                { label: 'AUC-PR',  value: run.metrics.auc_pr.toFixed(4),   good: run.metrics.auc_pr >= 0.89 },
                { label: 'AUC-ROC', value: run.metrics.auc_roc.toFixed(4),  good: true },
                { label: 'F1',      value: run.metrics.f1.toFixed(4),       good: run.metrics.f1 >= 0.5 },
                { label: 'FNR',     value: `${(run.metrics.fnr*100).toFixed(1)}%`, good: run.metrics.fnr <= 0.1 },
                { label: 'Recall',  value: `${(run.gates.recall_gate.value*100).toFixed(1)}%`, good: run.gates.recall_gate.passed },
                { label: 'Threshold', value: run.metrics.threshold_tuned, good: true },
              ].map(m => (
                <div key={m.label} className="bg-surface-3 rounded p-2 text-center">
                  <div className="text-[9px] text-text-muted mb-0.5">{m.label}</div>
                  <div className={`text-xs font-mono font-semibold ${m.good ? 'text-accent-green' : 'text-risk-critical'}`}>{m.value}</div>
                </div>
              ))}
            </div>

            {/* Gates */}
            <div className="space-y-0.5">
              <GateRow
                label="AUC-PR gate"
                threshold={`≥ ${run.gates.auc_pr_gate.threshold}`}
                value={run.gates.auc_pr_gate.value.toFixed(4)}
                passed={run.gates.auc_pr_gate.passed}
              />
              <GateRow
                label="Bias gate (FNR disparity)"
                threshold={`≤ ${run.gates.fnr_disparity_gate.threshold}`}
                value={run.gates.fnr_disparity_gate.value.toFixed(2)}
                passed={run.gates.fnr_disparity_gate.passed}
              />
              <GateRow
                label="Recall gate"
                threshold={`≥ ${run.gates.recall_gate.threshold}`}
                value={run.gates.recall_gate.value.toFixed(3)}
                passed={run.gates.recall_gate.passed}
              />
            </div>

            {/* Vertex AI */}
            <div className="mt-3 pt-2 border-t border-border-subtle">
              <div className="text-[10px] text-text-muted font-mono">
                vertex: {run.vertex_model} · <span className={run.vertex_stage === 'production' ? 'text-accent-green' : 'text-accent-orange'}>{run.vertex_stage}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Confusion matrix + PR curve */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
          <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">
            Production Run: 970bb676 (California)
          </div>
          <div className="grid grid-cols-2 gap-4">
            <ConfusionMatrix cm={OBJ1_RUNS[0].metrics.confusion_matrix} />
            <div>
              <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Best Hyperparameters</div>
              <div className="space-y-1">
                {Object.entries(OBJ1_RUNS[0].best_params).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-[10px] font-mono text-text-muted">{k}</span>
                    <span className="text-[10px] font-mono text-text-secondary">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* PR Curve */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
          <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-1">
            Precision-Recall Curve  <span className="text-text-muted font-normal normal-case">(CA · AUC-PR = 0.9051)</span>
          </div>
          <div className="text-text-muted text-[10px] font-mono mb-3">▲ tuned threshold: recall=0.903, precision=0.748</div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={PR_CURVE_CA} margin={{ top: 2, right: 8, left: -15, bottom: 2 }}>
              <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={{ fontSize: 9, fill: '#4a5978' }} label={{ value: 'Recall', position: 'bottom', fontSize: 9, fill: '#4a5978' }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: '#4a5978' }} label={{ value: 'Precision', angle: -90, position: 'left', fontSize: 9, fill: '#4a5978' }} />
              <Tooltip
                contentStyle={{ background: '#131b2e', border: '1px solid #253348', borderRadius: 4, fontSize: 10 }}
                formatter={(v) => [v.toFixed(3)]}
              />
              <ReferenceLine x={0.903} stroke="#f59e0b" strokeDasharray="3 3" strokeWidth={1} label={{ value: 'threshold', fontSize: 8, fill: '#f59e0b' }} />
              <Line type="monotone" dataKey="precision" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* SHAP importance */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-1">
          SHAP Feature Importance  <span className="text-text-muted font-normal normal-case">(mean |SHAP| · CA production run)</span>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={shapData} layout="vertical" margin={{ top: 2, right: 20, left: 10, bottom: 2 }}>
            <XAxis type="number" tick={{ fontSize: 9, fill: '#4a5978' }} axisLine={false} tickLine={false} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 9, fill: '#8a9bbf', fontFamily: 'monospace' }} width={160} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: '#131b2e', border: '1px solid #253348', borderRadius: 4, fontSize: 10 }}
              formatter={(v) => [v.toFixed(4), 'mean |SHAP|']}
            />
            <Bar dataKey="importance" radius={[0, 2, 2, 0]}>
              {shapData.map((entry, i) => (
                <Cell key={i} fill={i < 3 ? '#ef4444' : i < 6 ? '#f97316' : i < 9 ? '#f59e0b' : '#3b82f6'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Bias analysis */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider">
            Bias Analysis — False Negative Rate Disparity
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle className="w-3.5 h-3.5 text-accent-green" />
            <span className="text-[10px] font-mono text-accent-green">GATE PASS  (max disparity {BIAS_ANALYSIS.max_observed_disparity.toFixed(3)} &lt; {BIAS_ANALYSIS.max_disparity_threshold})</span>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {BIAS_ANALYSIS.slices.map(s => (
            <div key={s.group} className="flex items-center gap-2 bg-surface-3 border border-border-subtle rounded p-2">
              <CheckCircle className="w-3 h-3 text-accent-green flex-shrink-0" />
              <span className="text-[10px] text-text-secondary flex-1">{s.group}</span>
              <span className="text-[10px] font-mono text-text-muted">n={s.n}</span>
              <span className="text-[10px] font-mono font-semibold text-accent-green">FNR {(s.fnr*100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <div className="mt-2 text-[10px] text-text-muted font-mono">
          metric: {BIAS_ANALYSIS.metric}  ·  min_group_size: {BIAS_ANALYSIS.min_group_size}  ·  min_fire_count: {BIAS_ANALYSIS.min_fire_count}
        </div>
      </div>
    </div>
  );
}
