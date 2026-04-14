import { CheckCircle, XCircle, Clock, AlertTriangle, Database, ChevronDown, ChevronRight, Package } from 'lucide-react';
import { useState } from 'react';
import {
  PIPELINE_META, INGESTION_STAGES, PROCESSING_STAGES,
  FUSION_STAGE, VALIDATION_STAGE, EXPORT_STAGE, DATA_QUALITY_FLAGS
} from '../../data/mockPipelineData';

function StatusBadge({ status }) {
  const cfg = {
    success: { bg: 'bg-accent-green/10 border-accent-green/30 text-accent-green', label: 'PASS' },
    warning: { bg: 'bg-accent-orange/10 border-accent-orange/30 text-accent-orange', label: 'WARN' },
    failed:  { bg: 'bg-risk-critical/10 border-risk-critical/30 text-risk-critical', label: 'FAIL' },
    cached:  { bg: 'bg-text-muted/10 border-text-muted/30 text-text-muted', label: 'CACHED' },
    stub:    { bg: 'bg-text-muted/10 border-text-muted/30 text-text-muted', label: 'STUB' },
    pass:    { bg: 'bg-accent-green/10 border-accent-green/30 text-accent-green', label: 'PASS' },
    fail:    { bg: 'bg-risk-critical/10 border-risk-critical/30 text-risk-critical', label: 'FAIL' },
  };
  const c = cfg[status] ?? cfg.cached;
  return (
    <span className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border ${c.bg}`}>
      {c.label}
    </span>
  );
}

function StageHeader({ icon: Icon, number, title, sub, status, expanded, onToggle }) {
  return (
    <button
      onClick={onToggle}
      className="w-full flex items-center gap-3 p-3 hover:bg-surface-3/50 transition-colors rounded-t"
    >
      <div className="w-6 h-6 rounded bg-surface-3 border border-border-default flex items-center justify-center flex-shrink-0">
        <span className="text-[10px] font-mono text-text-muted">{number}</span>
      </div>
      {Icon && <Icon className="w-3.5 h-3.5 text-text-muted flex-shrink-0" />}
      <div className="flex-1 text-left">
        <div className="text-text-primary text-xs font-semibold">{title}</div>
        <div className="text-text-muted text-[10px] font-mono">{sub}</div>
      </div>
      <StatusBadge status={status} />
      {expanded ? <ChevronDown className="w-3 h-3 text-text-muted ml-1" /> : <ChevronRight className="w-3 h-3 text-text-muted ml-1" />}
    </button>
  );
}

function ColumnList({ columns }) {
  return (
    <div className="flex flex-wrap gap-1 mt-2">
      {columns.map(c => (
        <span key={c} className="text-[9px] font-mono bg-surface-3 border border-border-subtle text-text-secondary px-1.5 py-0.5 rounded">
          {c}
        </span>
      ))}
    </div>
  );
}

function KV({ label, value, mono = true }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-text-muted text-[10px] w-36 flex-shrink-0">{label}</span>
      <span className={`text-text-secondary text-[10px] ${mono ? 'font-mono' : ''} break-all`}>{value}</span>
    </div>
  );
}

export default function DataPipeline() {
  const [expanded, setExpanded] = useState({ ingest: true, process: false, fuse: false, validate: true, export: false });
  const toggle = (k) => setExpanded(e => ({ ...e, [k]: !e[k] }));

  // Pipeline arrow connector
  const Arrow = () => (
    <div className="flex justify-center py-1">
      <div className="w-px h-4 bg-border-default relative">
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 border-r border-b border-border-default rotate-45 translate-y-1" />
      </div>
    </div>
  );

  return (
    <div className="p-6 overflow-y-auto h-full">
      {/* DAG meta strip */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-3 mb-5 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">DAG</div>
            <div className="text-text-primary text-xs font-mono">{PIPELINE_META.dag_id}</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Schedule</div>
            <div className="text-text-secondary text-xs font-mono">{PIPELINE_META.schedule} UTC</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Regions</div>
            <div className="text-text-secondary text-xs font-mono">{PIPELINE_META.regions.join(' + ')}</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Grid Res</div>
            <div className="text-text-secondary text-xs font-mono">{PIPELINE_META.grid_resolution_km} km  (H3 res-2)</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Mode</div>
            <div className="text-accent-green text-xs font-mono font-semibold">{PIPELINE_META.operational_mode}</div>
          </div>
        </div>
        <div className="text-text-muted text-[10px] font-mono text-right">
          <div>Last run: 2025-01-15 18:04 UTC</div>
          <div>Next run: 2025-01-16 00:00 UTC</div>
        </div>
      </div>

      <div className="max-w-3xl mx-auto">

        {/* ─── STAGE 1: INGESTION ─────────────────────────────────────────────── */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg overflow-hidden">
          <StageHeader
            number="1" title="Ingestion" icon={Database}
            sub="ingest_firms · ingest_weather · ingest_landfire · ingest_srtm · ingest_goes"
            status="success"
            expanded={expanded.ingest}
            onToggle={() => toggle('ingest')}
          />
          {expanded.ingest && (
            <div className="px-4 pb-4 border-t border-border-subtle">
              <div className="mt-3 space-y-3">
                {INGESTION_STAGES.map(stage => (
                  <div key={stage.id} className="bg-surface-3 border border-border-subtle rounded p-3">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-text-primary text-xs font-semibold">{stage.label}</span>
                        <StatusBadge status={stage.status} />
                      </div>
                      <span className="text-text-muted text-[10px] font-mono">{stage.module}</span>
                    </div>
                    <div className="space-y-1">
                      <KV label="Source" value={stage.source} mono={false} />
                      <KV label="Output" value={stage.output_path} />
                      {stage.records_fetched != null && (
                        <KV label="Records fetched" value={String(stage.records_fetched)} />
                      )}
                      {stage.last_run && (
                        <KV label="Last run" value={stage.last_run} />
                      )}
                    </div>
                    {stage.key_columns.length > 0 && (
                      <>
                        <div className="text-text-muted text-[9px] uppercase tracking-wider mt-2 mb-1">Key columns</div>
                        <ColumnList columns={stage.key_columns} />
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <Arrow />

        {/* ─── STAGE 2: PROCESSING ────────────────────────────────────────────── */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg overflow-hidden">
          <StageHeader
            number="2" title="Processing"
            sub="process_firms · process_weather · process_static"
            status="success"
            expanded={expanded.process}
            onToggle={() => toggle('process')}
          />
          {expanded.process && (
            <div className="px-4 pb-4 border-t border-border-subtle">
              <div className="mt-3 space-y-3">
                {PROCESSING_STAGES.map(stage => (
                  <div key={stage.id} className="bg-surface-3 border border-border-subtle rounded p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-text-primary text-xs font-semibold">{stage.label}</span>
                      <StatusBadge status={stage.status} />
                    </div>
                    <div className="space-y-1">
                      <KV label="Module" value={stage.module} />
                      <KV label="Output" value={stage.output_path} />
                      {stage.input_rows && <KV label="Input rows" value={String(stage.input_rows)} />}
                      <KV label="Output rows" value={String(stage.output_rows)} />
                    </div>
                    <div className="text-text-muted text-[9px] uppercase tracking-wider mt-2 mb-1">Operations</div>
                    <div className="flex flex-wrap gap-1">
                      {stage.ops.map(op => (
                        <span key={op} className="text-[9px] font-mono bg-accent-blue/10 border border-accent-blue/20 text-accent-blue px-1.5 py-0.5 rounded">{op}</span>
                      ))}
                    </div>
                    <div className="text-text-muted text-[9px] uppercase tracking-wider mt-2 mb-1">Output columns</div>
                    <ColumnList columns={stage.output_columns} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <Arrow />

        {/* ─── STAGE 3: FUSION ────────────────────────────────────────────────── */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg overflow-hidden">
          <StageHeader
            number="3" title="Feature Fusion"
            sub="fuse_features.py  ·  32 columns  ·  55 H3 cells"
            status="success"
            expanded={expanded.fuse}
            onToggle={() => toggle('fuse')}
          />
          {expanded.fuse && (
            <div className="px-4 pb-4 border-t border-border-subtle">
              <div className="mt-3 grid grid-cols-2 gap-3">
                <div className="bg-surface-3 border border-border-subtle rounded p-3 space-y-1">
                  <KV label="Function" value={FUSION_STAGE.function} />
                  <KV label="Join strategy" value={FUSION_STAGE.join_strategy} mono={false} />
                  <KV label="Grid cells (total)" value={`${FUSION_STAGE.grid_cells_total}  (CA: ${FUSION_STAGE.grid_cells_ca}  TX: ${FUSION_STAGE.grid_cells_tx})`} />
                  <KV label="Fire cells" value={String(FUSION_STAGE.fire_cells)} />
                  <KV label="Output columns" value={String(FUSION_STAGE.output_columns_count)} />
                  <KV label="T-1 lag applied" value={FUSION_STAGE.temporal_lag_applied ? 'Yes (ML variant)' : 'No'} />
                </div>
                <div className="bg-surface-3 border border-border-subtle rounded p-3">
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Fill Strategies</div>
                  <div className="space-y-2">
                    <div>
                      <div className="text-[9px] text-accent-blue mb-1">forward_fill</div>
                      <ColumnList columns={FUSION_STAGE.fill_strategies.forward_fill} />
                    </div>
                    <div>
                      <div className="text-[9px] text-accent-orange mb-1">zero_fill</div>
                      <ColumnList columns={FUSION_STAGE.fill_strategies.zero_fill} />
                    </div>
                    <div>
                      <div className="text-[9px] text-text-muted mb-1">default_zero (non-fire cells)</div>
                      <ColumnList columns={FUSION_STAGE.fill_strategies.default_zero} />
                    </div>
                  </div>
                </div>
              </div>
              <div className="mt-3 bg-surface-3 border border-border-subtle rounded p-3">
                <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Output paths</div>
                <div className="space-y-1">
                  <KV label="Latest (raw)" value={FUSION_STAGE.output_path_latest} />
                  <KV label="Latest (ML)" value={FUSION_STAGE.output_path_ml} />
                  <KV label="Partitioned" value={FUSION_STAGE.output_path_partitioned} />
                </div>
              </div>
            </div>
          )}
        </div>

        <Arrow />

        {/* ─── STAGE 4: VALIDATION ────────────────────────────────────────────── */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg overflow-hidden">
          <StageHeader
            number="4" title="Validation"
            sub="validate_schema · detect_anomalies · bias_analysis"
            status="success"
            expanded={expanded.validate}
            onToggle={() => toggle('validate')}
          />
          {expanded.validate && (
            <div className="px-4 pb-4 border-t border-border-subtle">
              <div className="mt-3 grid grid-cols-2 gap-3">
                {/* Schema checks */}
                <div className="bg-surface-3 border border-border-subtle rounded p-3">
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Schema Checks ({VALIDATION_STAGE.checks_passed}/{VALIDATION_STAGE.checks_total})</div>
                  <div className="space-y-2">
                    {VALIDATION_STAGE.details.map(d => (
                      <div key={d.check} className="flex items-start gap-2">
                        {d.status === 'pass'
                          ? <CheckCircle className="w-3 h-3 text-accent-green flex-shrink-0 mt-0.5" />
                          : <XCircle className="w-3 h-3 text-risk-critical flex-shrink-0 mt-0.5" />
                        }
                        <div>
                          <div className="text-[10px] text-text-secondary font-mono">{d.check}</div>
                          <div className="text-[10px] text-text-muted">{d.detail}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                {/* Anomaly detection */}
                <div className="bg-surface-3 border border-border-subtle rounded p-3">
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Anomaly Detection</div>
                  <div className="space-y-1">
                    <KV label="Method" value={VALIDATION_STAGE.anomaly_detection.method} mono={false} />
                    <KV label="z-score (fire season)" value={`≥ ${VALIDATION_STAGE.anomaly_detection.threshold_fire_season}`} />
                    <KV label="z-score (off-season)" value={`≥ ${VALIDATION_STAGE.anomaly_detection.threshold_off_season}`} />
                    <KV label="Anomalies detected" value={String(VALIDATION_STAGE.anomaly_detection.anomalies_detected)} />
                    <KV label="Slack alert sent" value={VALIDATION_STAGE.anomaly_detection.slack_alert_sent ? 'YES' : 'no'} />
                  </div>
                  <div className="text-text-muted text-[9px] uppercase tracking-wider mt-3 mb-1">Monitored Features</div>
                  <ColumnList columns={VALIDATION_STAGE.anomaly_detection.monitored_features} />
                </div>
              </div>
            </div>
          )}
        </div>

        <Arrow />

        {/* ─── STAGE 5: EXPORT ────────────────────────────────────────────────── */}
        <div className="bg-surface-2 border border-border-subtle rounded-lg overflow-hidden">
          <StageHeader
            number="5" title="Export & Version"
            sub="export_spatial.py  ·  Parquet + NPZ  ·  DVC tracked  ·  GCS"
            status="success"
            expanded={expanded.export}
            onToggle={() => toggle('export')}
          />
          {expanded.export && (
            <div className="px-4 pb-4 border-t border-border-subtle">
              <div className="mt-3 bg-surface-3 border border-border-subtle rounded p-3">
                <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Output Artifacts</div>
                <div className="space-y-2">
                  {EXPORT_STAGE.outputs.map(o => (
                    <div key={o.path} className="flex items-center gap-2">
                      <Package className="w-3 h-3 text-text-muted flex-shrink-0" />
                      <span className="text-[9px] font-mono text-accent-blue bg-accent-blue/5 border border-accent-blue/20 rounded px-1">{o.type}</span>
                      <span className="text-[10px] font-mono text-text-secondary flex-1">{o.path}</span>
                      <span className="text-[10px] font-mono text-text-muted">{o.size_kb} KB</span>
                    </div>
                  ))}
                </div>
                <div className="mt-3 flex items-center gap-4 pt-2 border-t border-border-subtle">
                  <KV label="DVC tracked" value={EXPORT_STAGE.dvc_tracked ? 'Yes' : 'No'} />
                  <KV label="GCS bucket" value={EXPORT_STAGE.gcs_bucket} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Data quality flag legend */}
        <div className="mt-5 bg-surface-2 border border-border-subtle rounded-lg p-4">
          <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">Data Quality Flag Reference</div>
          <div className="grid grid-cols-3 gap-2">
            {DATA_QUALITY_FLAGS.map(f => (
              <div key={f.flag} className="flex items-start gap-2 bg-surface-3 border border-border-subtle rounded p-2">
                <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border flex-shrink-0 ${
                  f.flag <= 1 ? 'text-accent-green bg-accent-green/10 border-accent-green/30' :
                  f.flag === 2 || f.flag === 4 ? 'text-accent-orange bg-accent-orange/10 border-accent-orange/30' :
                  f.flag === 3 ? 'text-accent-blue bg-accent-blue/10 border-accent-blue/30' :
                  'text-risk-critical bg-risk-critical/10 border-risk-critical/30'
                }`}>F{f.flag}</span>
                <span className="text-[10px] text-text-muted">{f.label}</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
