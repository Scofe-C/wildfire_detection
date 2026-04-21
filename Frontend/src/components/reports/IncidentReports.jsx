import { useState } from 'react';
import { FileText, CheckCircle, AlertTriangle, Clock, ChevronDown, ChevronRight, Loader } from 'lucide-react';
import { MOCK_REPORTS } from '../../data/mockReports';
import useAPI from '../../hooks/useAPI';
import { apiUrl } from '../../api';

const fmt2 = (v) => (v != null && !isNaN(v)) ? Number(v).toFixed(2) : '—';
const fmt3 = (v) => (v != null && !isNaN(v)) ? Number(v).toFixed(3) : '—';

function ModeBadge({ mode }) {
  const cfg = {
    QUIET:     'bg-accent-green/10 border-accent-green/30 text-accent-green',
    ACTIVE:    'bg-accent-orange/10 border-accent-orange/30 text-accent-orange',
    EMERGENCY: 'bg-risk-critical/10 border-risk-critical/30 text-risk-critical',
  };
  return (
    <span className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded border ${cfg[mode] ?? cfg.QUIET}`}>{mode ?? 'QUIET'}</span>
  );
}


/** Fetch full report JSON from backend and display rich content */
function ReportCard({ report }) {
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);

  // mode derived from operating_mode or risk_level
  const mode = report.operating_mode ?? report.mode ?? (
    report.risk_level === 'CRITICAL' ? 'EMERGENCY' :
    report.risk_level === 'HIGH'     ? 'ACTIVE'    : 'QUIET'
  );

  const borderColor = {
    QUIET:     'border-accent-green/30',
    ACTIVE:    'border-accent-orange/30',
    EMERGENCY: 'border-risk-critical/40',
  }[mode] ?? 'border-border-subtle';

  const modeIcon = {
    QUIET:     <CheckCircle className="w-4 h-4 text-accent-green" />,
    ACTIVE:    <AlertTriangle className="w-4 h-4 text-accent-orange" />,
    EMERGENCY: <AlertTriangle className="w-4 h-4 text-risk-critical" />,
  }[mode];

  const handleExpand = async () => {
    const next = !expanded;
    setExpanded(next);
    if (next && !detail && !loading) {
      setLoading(true);
      try {
        const res = await fetch(apiUrl(`/api/reports/${report.report_id || report.id}`));
        if (res.ok) setDetail(await res.json());
      } catch (_) {}
      setLoading(false);
    }
  };

  const conf = detail?.report_confidence ?? report.confidence ?? 0;
  const sources = Array.isArray(detail?.grounding_sources)
    ? detail.grounding_sources.length
    : (report.grounding_sources ?? '—');
  return (
    <div className={`bg-surface-2 border ${borderColor} rounded-lg overflow-hidden`}>
      {/* Card header */}
      <button
        onClick={handleExpand}
        className="w-full p-4 flex items-start gap-3 hover:bg-surface-3/40 transition-colors text-left"
      >
        <div className="flex-shrink-0 mt-0.5">{modeIcon}</div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <ModeBadge mode={mode} />
            <span className="text-[9px] font-mono bg-surface-3 border border-border-subtle text-text-muted px-1.5 py-0.5 rounded">
              {report.report_type ?? report.schema_type}
            </span>
            <span className="text-[9px] font-mono text-text-muted">{report.report_id ?? report.id}</span>
          </div>
          <div className="text-text-primary text-sm font-semibold mb-1">{report.title}</div>
          <div className="flex items-center gap-3 text-[10px] text-text-muted font-mono">
            <span><Clock className="w-2.5 h-2.5 inline mr-0.5" />{(report.generated_at ?? '').slice(0, 19).replace('T', ' ')}</span>
            <span>conf: <span className={conf >= 0.70 ? 'text-accent-green' : 'text-risk-critical'}>{fmt2(conf)}</span></span>
            <span>sources: {sources}</span>
            {report.human_review_required && (
              <span className="text-accent-orange">review required</span>
            )}
          </div>
        </div>
        {expanded ? <ChevronDown className="w-4 h-4 text-text-muted flex-shrink-0 mt-1" /> : <ChevronRight className="w-4 h-4 text-text-muted flex-shrink-0 mt-1" />}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-border-subtle px-4 pb-4 pt-3 space-y-4">
          {loading && (
            <div className="flex items-center gap-2 text-text-muted text-xs">
              <Loader className="w-3 h-3 animate-spin" /> Loading report details…
            </div>
          )}

          {detail ? (
            <>
              {/* Incident header — name + status */}
              {detail.incident_name && (
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-text-primary text-sm font-semibold">{detail.incident_name}</span>
                  {detail.incident_status && (
                    <span className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded border
                      ${detail.incident_status === 'ACTIVE' ? 'text-risk-critical border-risk-critical/30 bg-risk-critical/10' : 'text-accent-green border-accent-green/30 bg-accent-green/10'}`}>
                      {detail.incident_status}
                    </span>
                  )}
                  {detail.percent_contained != null && (
                    <span className="text-[10px] font-mono text-text-muted">{detail.percent_contained}% contained</span>
                  )}
                </div>
              )}

              {/* Spread summary */}
              <div>
                <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">
                  {detail.spread_summary ? 'Spread Summary' : 'Risk Summary'}
                </div>
                <p className="text-text-secondary text-[11px] leading-relaxed bg-surface-3 border border-border-subtle rounded p-3">
                  {detail.spread_summary || detail.risk_summary || '—'}
                </p>
              </div>

              {/* Weather + Fire behavior row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {detail.weather_observations && (
                  <div>
                    <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Weather</div>
                    <div className="flex flex-wrap gap-3 bg-surface-3 border border-border-subtle rounded p-3">
                      {[
                        { label: 'Temp', val: detail.weather_observations.temperature_f != null ? `${fmt2(detail.weather_observations.temperature_f)} °F` : null },
                        { label: 'RH',   val: detail.weather_observations.relative_humidity_pct != null ? `${fmt2(detail.weather_observations.relative_humidity_pct)} %` : null },
                        { label: 'Wind', val: detail.weather_observations.wind_speed_mph != null ? `${fmt2(detail.weather_observations.wind_speed_mph)} mph` : null },
                      ].filter(x => x.val).map(({ label, val }) => (
                        <div key={label} className="text-center">
                          <div className="text-[9px] text-text-muted font-mono">{label}</div>
                          <div className="text-xs font-mono text-text-primary">{val}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {detail.fire_behavior && (
                  <div>
                    <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Fire Behavior</div>
                    <div className="flex flex-wrap gap-3 bg-surface-3 border border-border-subtle rounded p-3">
                      {[
                        { label: 'ROS',   val: detail.fire_behavior.rate_of_spread },
                        { label: 'Flame', val: detail.fire_behavior.flame_length_ft != null ? `${detail.fire_behavior.flame_length_ft} ft` : null },
                        { label: 'Type',  val: detail.fire_behavior.fire_type },
                      ].filter(x => x.val).map(({ label, val }) => (
                        <div key={label} className="text-center">
                          <div className="text-[9px] text-text-muted font-mono">{label}</div>
                          <div className="text-xs font-mono text-text-primary">{val}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Immediate actions */}
              {detail.immediate_actions?.length > 0 && (
                <div>
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Immediate Actions</div>
                  <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-2">
                    {detail.immediate_actions.map((action, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <span className="text-[9px] font-mono font-bold text-risk-critical flex-shrink-0 mt-0.5">{String(i + 1).padStart(2, '0')}</span>
                        <span className="text-[10px] text-text-secondary leading-tight">{typeof action === 'string' ? action : action.description ?? action.title ?? JSON.stringify(action)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Also handle non-incident: preventive_recommendations */}
              {!detail.immediate_actions?.length && detail.preventive_recommendations?.length > 0 && (
                <div>
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Recommended Actions</div>
                  <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-2">
                    {detail.preventive_recommendations.map((rec, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <span className={`text-[9px] font-mono font-bold flex-shrink-0 px-1 rounded ${rec.priority === 'CRITICAL' ? 'text-risk-critical' : rec.priority === 'HIGH' ? 'text-risk-high' : 'text-text-muted'}`}>
                          {rec.priority}
                        </span>
                        <div>
                          <div className="text-[10px] font-semibold text-text-primary">{rec.title}</div>
                          <div className="text-[10px] text-text-secondary leading-tight mt-0.5">{rec.description}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Communities + Evacuation row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {detail.affected_communities?.length > 0 && (
                  <div>
                    <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Affected Communities</div>
                    <div className="bg-surface-3 border border-border-subtle rounded p-3 flex flex-wrap gap-1.5">
                      {detail.affected_communities.map((c, i) => (
                        <span key={i} className="text-[10px] font-mono text-text-primary bg-surface-2 border border-border-subtle px-2 py-0.5 rounded">{c}</span>
                      ))}
                    </div>
                  </div>
                )}
                {detail.evacuation_status?.length > 0 && (
                  <div>
                    <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Evacuation Status</div>
                    <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-1.5">
                      {detail.evacuation_status.map((z, i) => (
                        <div key={i} className="flex items-center justify-between gap-2">
                          <span className="text-[10px] text-text-secondary leading-tight flex-1">{z.zone_name}</span>
                          <span className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded ${z.status === 'ORDER' ? 'text-risk-critical bg-risk-critical/10' : 'text-accent-orange bg-accent-orange/10'}`}>{z.status}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Resources + Projected losses */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {detail.resource_requirements?.length > 0 && (
                  <div>
                    <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Resource Requirements</div>
                    <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-1.5">
                      {detail.resource_requirements.map((r, i) => (
                        <div key={i} className="flex items-center justify-between">
                          <span className="text-[10px] font-mono text-text-secondary">{r.quantity}x {r.resource_type}</span>
                          <span className="text-[9px] font-mono text-text-muted">{r.ics_type}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {detail.projected_losses && (
                  <div>
                    <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Projected Impact</div>
                    <div className="flex flex-wrap gap-3 bg-surface-3 border border-border-subtle rounded p-3">
                      {[
                        { label: 'Structures', val: detail.projected_losses.structures_at_risk },
                        { label: 'Population', val: detail.projected_losses.population_at_risk },
                      ].filter(x => x.val != null).map(({ label, val }) => (
                        <div key={label} className="text-center">
                          <div className="text-lg font-mono font-bold text-risk-high">{val?.toLocaleString()}</div>
                          <div className="text-[9px] text-text-muted font-mono">{label} at risk</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Strategic objectives */}
              {detail.strategic_objectives?.length > 0 && (
                <div>
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Strategic Objectives</div>
                  <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-1">
                    {detail.strategic_objectives.map((obj, i) => (
                      <div key={i} className="flex items-start gap-1.5">
                        <div className="w-1 h-1 rounded-full bg-accent-blue flex-shrink-0 mt-1.5" />
                        <span className="text-[10px] font-mono text-text-secondary leading-tight">{obj}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Projected activity */}
              {detail.projected_activity && (
                <div>
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Projected Activity</div>
                  <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-2">
                    {['hours_12', 'hours_24', 'hours_48', 'hours_72'].map(k => {
                      const val = detail.projected_activity[k];
                      if (!val) return null;
                      return (
                        <div key={k} className="flex items-start gap-2">
                          <span className="text-[9px] font-mono font-bold text-text-muted flex-shrink-0 w-8">{k.replace('hours_', '')}h</span>
                          <span className="text-[10px] text-text-secondary leading-tight">{val}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Also handle non-incident fields */}
              {detail.top_risk_cells?.length > 0 && (
                <div>
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Top Risk Cells</div>
                  <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-1">
                    {detail.top_risk_cells.slice(0, 5).map((cell, i) => (
                      <div key={i} className="flex items-center justify-between">
                        <span className="text-[10px] font-mono text-text-muted">{cell.h3_index}</span>
                        <span className="text-[10px] font-mono font-bold text-risk-critical">{fmt3(cell.risk_score)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {detail.escalation_trigger && (
                <div>
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Escalation Trigger</div>
                  <p className="text-text-secondary text-[11px] leading-relaxed bg-surface-3 border border-border-subtle rounded p-3">
                    {detail.escalation_trigger}
                  </p>
                </div>
              )}

              {/* Footer */}
              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-border-subtle">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[9px] font-mono text-text-muted">type: {detail.report_type}</span>
                  {detail.disclaimer && (
                    <>
                      <span className="text-[9px] font-mono text-text-muted">·</span>
                      <span className="text-[9px] font-mono text-accent-orange">{detail.disclaimer}</span>
                    </>
                  )}
                </div>
                <div className="flex flex-wrap gap-2">
                  {detail.data_completeness && Object.entries(detail.data_completeness).map(([k, v]) => (
                    <span key={k} className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${v ? 'text-accent-green border-accent-green/30' : 'text-text-muted border-border-subtle'}`}>
                      {k}
                    </span>
                  ))}
                </div>
              </div>
            </>
          ) : !loading && (
            <div className="text-center text-text-muted text-xs py-4">
              Failed to load report details. Click to retry.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function IncidentReports() {
  const [modeFilter, setModeFilter] = useState('all');
  const { data: liveReports } = useAPI('/api/reports?limit=500');

  // Normalize list items to a stable card shape
  const reports = (liveReports && liveReports.length > 0)
    ? liveReports.map(r => ({
        report_id: r.id,
        id:         r.id,
        mode:       r.operating_mode ?? (
          r.risk_level === 'CRITICAL' ? 'EMERGENCY' :
          r.risk_level === 'HIGH'     ? 'ACTIVE'    : 'QUIET'
        ),
        risk_level:            r.risk_level,
        report_type:           r.report_type,
        schema_type:           r.report_type,
        generated_at:          r.generated_at,
        confidence:            r.confidence ?? r.report_confidence ?? 0,
        grounding_sources:     r.grounding_sources,
        human_review_required: r.human_review_required,
        title: r.title || `${(r.report_type || 'report').replace(/_/g, ' ')} — ${r.id}`,
        operating_mode: r.operating_mode,
        content: null, // loaded on expand from /api/reports/{id}
      }))
    : MOCK_REPORTS;

  const filtered = modeFilter === 'all'
    ? reports
    : reports.filter(r => r.mode === modeFilter);

  return (
    <div className="p-6 overflow-y-auto h-full">

      {/* Header strip */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-3 mb-5 flex flex-col md:flex-row items-start md:items-center gap-3 md:gap-0 justify-between">
        <div className="grid grid-cols-2 md:flex md:items-center gap-3 md:gap-6">
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
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Reports</div>
            <div className="text-text-secondary text-xs font-mono">{reports.length} generated</div>
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
            { name: 'BaseReport',     file: 'base_schema.py',      desc: 'Pydantic base' },
            { name: 'IncidentReport', file: 'incident_schema.py',  desc: 'ICS-209 aligned' },
            { name: 'DailyReport',    file: 'daily_schema.py',     desc: 'Daily briefing' },
            { name: 'HighRiskReport', file: 'high_risk_schema.py', desc: 'CRITICAL conditions' },
            { name: 'FinalReport',    file: 'final_schema.py',     desc: 'Post-incident' },
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

      {/* Validation config */}
      <div className="mt-4 bg-surface-2 border border-border-subtle rounded-lg p-3">
        <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Validation Requirements</div>
        <div className="flex flex-wrap items-center gap-3 md:gap-6 text-[10px] text-text-muted font-mono">
          <span>confidence_threshold: 0.70</span>
          <span>min_grounding_sources: 3</span>
          <span>section_completeness: required</span>
          <span>schema_validation: Pydantic strict</span>
        </div>
      </div>
    </div>
  );
}
