import { useState } from 'react';
import { FileText, CheckCircle, AlertTriangle, Clock, ChevronDown, ChevronRight } from 'lucide-react';
import { MOCK_REPORTS } from '../../data/mockReports';

function ModeBadge({ mode }) {
  const cfg = {
    QUIET:     'bg-accent-green/10 border-accent-green/30 text-accent-green',
    ACTIVE:    'bg-accent-orange/10 border-accent-orange/30 text-accent-orange',
    EMERGENCY: 'bg-risk-critical/10 border-risk-critical/30 text-risk-critical',
  };
  return (
    <span className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded border ${cfg[mode] ?? cfg.QUIET}`}>{mode}</span>
  );
}

function RiskCountBadge({ label, count, tier }) {
  const colors = {
    critical: 'text-risk-critical',
    high:     'text-risk-high',
    medium:   'text-risk-medium',
    low:      'text-risk-low',
  };
  return (
    <div className="text-center">
      <div className={`text-lg font-mono font-bold ${colors[tier]}`}>{count}</div>
      <div className="text-[9px] text-text-muted font-mono uppercase">{label}</div>
    </div>
  );
}

function ReportCard({ report }) {
  const [expanded, setExpanded] = useState(false);
  const rs = report.content.risk_summary;

  const modeIcon = {
    QUIET:     <CheckCircle className="w-4 h-4 text-accent-green" />,
    ACTIVE:    <AlertTriangle className="w-4 h-4 text-accent-orange" />,
    EMERGENCY: <AlertTriangle className="w-4 h-4 text-risk-critical" />,
  }[report.mode];

  const borderColor = {
    QUIET:     'border-accent-green/30',
    ACTIVE:    'border-accent-orange/30',
    EMERGENCY: 'border-risk-critical/40',
  }[report.mode];

  return (
    <div className={`bg-surface-2 border ${borderColor} rounded-lg overflow-hidden`}>
      {/* Card header */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full p-4 flex items-start gap-3 hover:bg-surface-3/40 transition-colors text-left"
      >
        <div className="flex-shrink-0 mt-0.5">{modeIcon}</div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <ModeBadge mode={report.mode} />
            <span className="text-[9px] font-mono bg-surface-3 border border-border-subtle text-text-muted px-1.5 py-0.5 rounded">{report.schema_type}</span>
            <span className="text-[9px] font-mono text-text-muted">{report.region}</span>
            <span className="text-[9px] font-mono text-text-muted">{report.report_id}</span>
          </div>
          <div className="text-text-primary text-sm font-semibold mb-1">{report.title}</div>
          <div className="flex items-center gap-3 text-[10px] text-text-muted font-mono">
            <span><Clock className="w-2.5 h-2.5 inline mr-0.5" />{report.generated_at}</span>
            <span>{report.llm_model}</span>
            <span>conf: <span className={report.confidence >= 0.70 ? 'text-accent-green' : 'text-risk-critical'}>{report.confidence.toFixed(2)}</span></span>
            <span>sources: {report.grounding_sources}</span>
          </div>
        </div>
        {/* Risk summary mini */}
        <div className="flex items-center gap-3 flex-shrink-0 mr-2">
          <RiskCountBadge label="CRIT" count={rs.critical_cells} tier="critical" />
          <RiskCountBadge label="HIGH" count={rs.high_cells} tier="high" />
          <RiskCountBadge label="MED"  count={rs.medium_cells} tier="medium" />
          <RiskCountBadge label="LOW"  count={rs.low_cells} tier="low" />
        </div>
        {expanded ? <ChevronDown className="w-4 h-4 text-text-muted flex-shrink-0 mt-1" /> : <ChevronRight className="w-4 h-4 text-text-muted flex-shrink-0 mt-1" />}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-border-subtle px-4 pb-4 pt-3 space-y-4">
          {/* Situation summary */}
          <div>
            <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Situation Summary</div>
            <p className="text-text-secondary text-[11px] leading-relaxed bg-surface-3 border border-border-subtle rounded p-3">
              {report.content.situation_summary}
            </p>
          </div>

          {/* Weather outlook */}
          <div>
            <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Weather Outlook</div>
            <p className="text-text-secondary text-[11px] leading-relaxed bg-surface-3 border border-border-subtle rounded p-3">
              {report.content.weather_outlook}
            </p>
          </div>

          {/* Driving features + actions */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Key Features Driving Risk</div>
              <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-1">
                {report.content.key_features_driving_risk.map((f, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <div className="w-1 h-1 rounded-full bg-risk-high flex-shrink-0" />
                    <span className="text-[10px] font-mono text-text-secondary">{f}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Recommended Actions</div>
              <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-1.5">
                {report.content.recommended_actions.map((a, i) => (
                  <div key={i} className="flex items-start gap-1.5">
                    <span className="text-[9px] font-mono text-text-muted flex-shrink-0 mt-0.5">{String(i + 1).padStart(2, '0')}</span>
                    <span className="text-[10px] text-text-secondary leading-tight">{a}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Risk grid */}
          <div className="grid grid-cols-5 gap-2">
            <div className="col-span-2 bg-surface-3 border border-border-subtle rounded p-3">
              <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Risk Distribution</div>
              <div className="flex items-center gap-3">
                <RiskCountBadge label="CRITICAL" count={rs.critical_cells} tier="critical" />
                <RiskCountBadge label="HIGH"     count={rs.high_cells}     tier="high" />
                <RiskCountBadge label="MEDIUM"   count={rs.medium_cells}   tier="medium" />
                <RiskCountBadge label="LOW"      count={rs.low_cells}      tier="low" />
              </div>
            </div>
            <div className="col-span-3 bg-surface-3 border border-border-subtle rounded p-3">
              <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Highest Risk Cell</div>
              <div className="flex items-center gap-3">
                <div>
                  <div className="text-text-primary text-xs font-semibold">{rs.highest_risk_cell}</div>
                  <div className="text-2xl font-mono font-bold text-risk-critical">{rs.highest_risk_score.toFixed(3)}</div>
                </div>
              </div>
              <div className="text-[9px] text-text-muted font-mono mt-1">{report.content.model_attribution}</div>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between pt-2 border-t border-border-subtle">
            <div className="flex items-center gap-2">
              <span className="text-[9px] font-mono text-text-muted">schema: {report.schema_type} (Pydantic/ICS-209)</span>
              <span className="text-[9px] font-mono text-text-muted">·</span>
              <span className="text-[9px] font-mono text-text-muted">grounding: {report.grounding_sources} sources ≥ min({3})</span>
            </div>
            <div className="flex gap-2">
              {['JSON', 'Markdown', 'HTML'].map(fmt => (
                <span key={fmt} className="text-[9px] font-mono text-text-muted border border-border-subtle rounded px-1.5 py-0.5 bg-surface-3">
                  {fmt}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function IncidentReports() {
  const [modeFilter, setModeFilter] = useState('all');

  const filtered = modeFilter === 'all'
    ? MOCK_REPORTS
    : MOCK_REPORTS.filter(r => r.mode === modeFilter);

  return (
    <div className="p-6 overflow-y-auto h-full">

      {/* Header strip */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-3 mb-5 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Generator</div>
            <div className="text-text-primary text-xs font-semibold">GeminiDisasterReporter (OBJ-3)</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Schema</div>
            <div className="text-text-secondary text-xs font-mono">ICS-209 aligned (Pydantic)</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">LLM Backend</div>
            <div className="text-text-secondary text-xs font-mono">gemini-2.5-flash (Vertex AI)</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Today</div>
            <div className="text-text-secondary text-xs font-mono">2 reports generated</div>
          </div>
        </div>

        {/* Mode filter */}
        <div className="flex items-center gap-1">
          {['all', 'QUIET', 'ACTIVE', 'EMERGENCY'].map(m => {
            const active = modeFilter === m;
            const colors = {
              all:       active ? 'bg-surface-3 border-border-default text-text-primary' : 'border-border-subtle text-text-muted',
              QUIET:     active ? 'bg-accent-green/20 border-accent-green/40 text-accent-green' : 'border-border-subtle text-text-muted',
              ACTIVE:    active ? 'bg-accent-orange/20 border-accent-orange/40 text-accent-orange' : 'border-border-subtle text-text-muted',
              EMERGENCY: active ? 'bg-risk-critical/20 border-risk-critical/40 text-risk-critical' : 'border-border-subtle text-text-muted',
            };
            return (
              <button
                key={m}
                onClick={() => setModeFilter(m)}
                className={`text-[10px] font-mono px-2.5 py-1 rounded border transition-colors hover:bg-surface-3 ${colors[m]}`}
              >
                {m}
              </button>
            );
          })}
        </div>
      </div>

      {/* Report schema reference */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-3 mb-5">
        <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Report Schema Hierarchy (src/models/obj3_gemini/schemas/)</div>
        <div className="flex items-center gap-2 flex-wrap">
          {[
            { name: 'BaseReport',    file: 'base_schema.py',        desc: 'Pydantic base' },
            { name: 'IncidentReport', file: 'incident_schema.py',   desc: 'ICS-209 aligned' },
            { name: 'DailyReport',   file: 'daily_schema.py',       desc: 'Daily briefing' },
            { name: 'HighRiskReport',file: 'high_risk_schema.py',   desc: 'CRITICAL conditions' },
            { name: 'FinalReport',   file: 'final_schema.py',       desc: 'Post-incident' },
          ].map(s => (
            <div key={s.name} className="flex items-center gap-1.5 bg-surface-3 border border-border-subtle rounded px-2 py-1.5">
              <FileText className="w-3 h-3 text-text-muted" />
              <div>
                <div className="text-[10px] font-mono text-text-primary">{s.name}</div>
                <div className="text-[9px] text-text-muted">{s.file}  ·  {s.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Reports list */}
      <div className="space-y-3">
        {filtered.length === 0 ? (
          <div className="text-center text-text-muted text-xs py-8">No reports for selected mode filter.</div>
        ) : (
          filtered.map(report => <ReportCard key={report.report_id} report={report} />)
        )}
      </div>

      {/* Reporting config note */}
      <div className="mt-4 bg-surface-2 border border-border-subtle rounded-lg p-3">
        <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Validation Requirements</div>
        <div className="flex items-center gap-6 text-[10px] text-text-muted font-mono">
          <span>confidence_threshold: 0.70</span>
          <span>min_grounding_sources: 3</span>
          <span>section_completeness: required</span>
          <span>schema_validation: Pydantic strict</span>
        </div>
      </div>
    </div>
  );
}
