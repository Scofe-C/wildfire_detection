import { CheckCircle, AlertTriangle, MessageSquare } from 'lucide-react';
import { OBJ3_STATE, MODE_MATRIX, WATCHDOG_CONFIG } from '../../data/mockModelData';

function KV({ label, value, mono = true, color }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-text-muted text-[10px] w-44 flex-shrink-0">{label}</span>
      <span className={`text-[10px] ${mono ? 'font-mono' : ''} ${color ?? 'text-text-secondary'}`}>{value}</span>
    </div>
  );
}

function ModeBadge({ mode }) {
  const cfg = {
    QUIET:     'bg-accent-green/10 border-accent-green/30 text-accent-green',
    ACTIVE:    'bg-accent-orange/10 border-accent-orange/30 text-accent-orange',
    EMERGENCY: 'bg-risk-critical/10 border-risk-critical/30 text-risk-critical',
  };
  return (
    <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${cfg[mode] ?? cfg.QUIET}`}>{mode}</span>
  );
}

export default function OBJ3Reporter() {
  const s = OBJ3_STATE;
  const wc = WATCHDOG_CONFIG;
  const currentMode = wc.modes[wc.current_mode];

  return (
    <div className="p-6 overflow-y-auto h-full space-y-5">

      {/* Header strip */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Objective</div>
            <div className="text-text-primary text-xs font-semibold">OBJ-3: AI Disaster Report Orchestrator</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Class</div>
            <div className="text-text-secondary text-xs font-mono">GeminiDisasterReporter</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">LLM Backend</div>
            <div className="text-text-secondary text-xs font-mono">{s.llm_model}</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Current Mode</div>
            <ModeBadge mode={s.operational_mode} />
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Confidence Gate</div>
            <div className="text-accent-green text-xs font-mono font-semibold">≥ {s.confidence_threshold}</div>
          </div>
        </div>
      </div>

      {/* Current state panel */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-surface-2 border border-accent-green/30 rounded-lg p-4">
          <div className="text-text-muted text-[9px] uppercase tracking-wider mb-3">Current State</div>
          <div className="space-y-2">
            <KV label="operational_mode" value={s.operational_mode} color="text-accent-green" />
            <KV label="emergency_sub_state" value={s.emergency_sub_state ?? 'null'} />
            <KV label="risk_level" value={s.risk_level} />
            <KV label="firms_hotspot_count" value={String(s.firms_hotspot_count)} />
            <KV label="is_deployable" value={String(s.is_deployable)} />
            <KV label="mode_disagreement" value={String(s.mode_disagreement)} />
            <KV label="reports_today" value={String(s.reports_generated_today)} />
            <KV label="last_report_at" value={s.last_report_at} />
          </div>
        </div>

        <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
          <div className="text-text-muted text-[9px] uppercase tracking-wider mb-3">LLM Configuration</div>
          <div className="space-y-2">
            <KV label="llm_backend" value={s.llm_backend} />
            <KV label="llm_model" value={s.llm_model} />
            <KV label="confidence_threshold" value={String(s.confidence_threshold)} />
            <KV label="min_grounding_sources" value={String(s.min_grounding_sources)} />
            <KV label="corpus_chars_loaded" value={`${s.corpus_chars_loaded.toLocaleString()} / ${s.corpus_max_chars.toLocaleString()}`} />
            <KV label="last_report_confidence" value={s.report_confidence.toFixed(2)} color={s.report_confidence >= s.confidence_threshold ? 'text-accent-green' : 'text-risk-critical'} />
          </div>

          <div className="mt-3 pt-2 border-t border-border-subtle">
            <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">LLM Backends (swappable)</div>
            {[
              { id: 'ollama',      label: 'Ollama (local)',     model: 'qwen3:8b',          phase: '1', active: false },
              { id: 'gemini_dev',  label: 'Gemini Dev API',     model: 'gemini-2.5-flash',   phase: '2', active: true },
              { id: 'vertex_ai',   label: 'Vertex AI',          model: 'Full project',       phase: '3', active: false },
            ].map(b => (
              <div key={b.id} className={`flex items-center gap-2 py-1 ${b.active ? 'opacity-100' : 'opacity-40'}`}>
                <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${b.active ? 'bg-accent-green' : 'bg-border-default'}`} />
                <span className="text-[10px] text-text-secondary font-mono">{b.label}</span>
                <span className="text-[10px] text-text-muted font-mono">{b.model}</span>
                <span className="text-[9px] text-text-muted ml-auto">Phase {b.phase}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
          <div className="text-text-muted text-[9px] uppercase tracking-wider mb-3">Report Output Formats</div>
          {[
            { mode: 'QUIET',     type: 'incident_brief',      schema: 'IncidentReport' },
            { mode: 'ACTIVE',    type: 'tactical_operations', schema: 'DailyReport' },
            { mode: 'EMERGENCY', type: 'strategic_impact',    schema: 'HighRiskReport' },
            { mode: 'POST_FIRE', type: 'lessons_learned',     schema: 'FinalReport' },
          ].map(r => (
            <div key={r.mode} className="flex items-center gap-2 py-1.5 border-b border-border-subtle/50">
              <ModeBadge mode={r.mode === 'POST_FIRE' ? 'ACTIVE' : r.mode} />
              <span className="text-[10px] font-mono text-text-secondary flex-1">{r.type}</span>
              <span className="text-[9px] text-text-muted font-mono">{r.schema}</span>
            </div>
          ))}
          <div className="mt-3 text-text-muted text-[9px]">Output formats: JSON · Markdown · HTML (Jinja2)</div>
          <div className="mt-1 text-text-muted text-[9px]">Schema: ICS-209 aligned (Pydantic)</div>
        </div>
      </div>

      {/* Mode resolution state machine */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">
          State Machine — Mode Resolution Matrix
          <span className="text-text-muted font-normal normal-case ml-2">(src/models/obj3_gemini/state_machine.py)</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-border-subtle">
                {['risk_level', 'firms_hotspot_count', 'is_deployable', '→ mode', 'disagreement'].map(col => (
                  <th key={col} className="text-left text-text-muted font-mono py-1.5 pr-4 uppercase text-[9px] tracking-wider">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MODE_MATRIX.map((row, i) => (
                <tr key={i} className={`border-b border-border-subtle/50 ${i === 0 ? 'bg-accent-green/5' : ''}`}>
                  <td className="py-1.5 pr-4 font-mono text-text-secondary">{row.risk}</td>
                  <td className="py-1.5 pr-4 font-mono text-text-secondary">{row.hotspots}</td>
                  <td className="py-1.5 pr-4 font-mono text-text-secondary">{String(row.deployable)}</td>
                  <td className="py-1.5 pr-4"><ModeBadge mode={row.mode} /></td>
                  <td className={`py-1.5 pr-4 font-mono text-[10px] ${row.disagreement ? 'text-accent-orange' : 'text-text-muted'}`}>
                    {row.disagreement ? 'YES ⚠' : 'false'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 text-[10px] text-text-muted font-mono">
          Current: risk_level={s.risk_level}  firms_hotspot_count={s.firms_hotspot_count}  is_deployable={String(s.is_deployable)}  → <span className="text-accent-green font-semibold">QUIET</span>
        </div>
      </div>

      {/* Watchdog configuration */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">
          Watchdog Configuration  <span className="text-text-muted font-normal normal-case">(schema_config.yaml)</span>
        </div>
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(wc.modes).map(([name, cfg]) => (
            <div key={name} className={`bg-surface-3 border rounded p-3 ${name === wc.current_mode ? 'border-accent-green/40' : 'border-border-subtle'}`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-text-primary text-xs font-mono font-semibold uppercase">{name}</span>
                {name === wc.current_mode && (
                  <span className="text-[9px] font-mono text-accent-green border border-accent-green/30 bg-accent-green/10 px-1.5 py-0.5 rounded">ACTIVE</span>
                )}
              </div>
              <div className="space-y-1">
                <KV label="poll_interval_min" value={`${cfg.poll_interval_min} min`} />
                <KV label="pipeline_interval_hr" value={`${cfg.pipeline_interval_hr} hr`} />
                <KV label="resolution_km" value={`${cfg.resolution_km} km`} />
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-4 mt-4">
          <div className="bg-surface-3 border border-border-subtle rounded p-3">
            <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">False Alarm Gates</div>
            <div className="space-y-1">
              {Object.entries(wc.false_alarm_gates).map(([k, v]) => (
                <KV key={k} label={k} value={String(v)} />
              ))}
            </div>
          </div>
          <div className="bg-surface-3 border border-border-subtle rounded p-3">
            <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Emergency Trigger</div>
            <div className="space-y-1">
              {Object.entries(wc.emergency_trigger).map(([k, v]) => (
                <KV key={k} label={k} value={String(v)} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
