import { useState, useEffect, useCallback, useRef } from 'react';
import {
  CheckCircle, AlertTriangle, MessageSquare, RefreshCw, Play, ExternalLink,
  Trash2, Eye, FileText, Zap, XCircle, Send, Search, ArrowUpDown,
  Upload, Copy, ChevronDown, Loader2, BarChart3, Shield, Clock,
} from 'lucide-react';
import Badge from '../ui/Badge';
import Card from '../ui/Card';
import Section from '../ui/Section';
import Spinner from '../ui/Spinner';
import useAPI from '../../hooks/useAPI';
import { apiUrl } from '../../api';
import { MODE_MATRIX, WATCHDOG_CONFIG } from '../../data/mockModelData';

// ─── Constants ────────────────────────────────────────────────────────────────
const RISK_LEVELS = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'];
const REPORT_TYPES = ['all', 'daily', 'high_risk', 'incident', 'final'];

const PRESETS = {
  low:       { risk_level: 'LOW',      firms_hotspot_count: 0,  temperature_max: 72,  wind_speed_mph: 8,  relative_humidity: 65, soil_moisture: 0.28 },
  high:      { risk_level: 'HIGH',     firms_hotspot_count: 4,  temperature_max: 101, wind_speed_mph: 22, relative_humidity: 14, soil_moisture: 0.06 },
  emergency: { risk_level: 'CRITICAL', firms_hotspot_count: 15, temperature_max: 109, wind_speed_mph: 38, relative_humidity: 6,  soil_moisture: 0.03 },
};

const EDIT_SECTIONS = {
  incident: [
    { key: 'situation_summary',     type: 'textarea', label: 'Situation Summary' },
    { key: 'weather_conditions',    type: 'textarea', label: 'Weather Conditions' },
    { key: 'fire_behavior',         type: 'textarea', label: 'Fire Behavior' },
    { key: 'current_actions',       type: 'textarea', label: 'Current Actions' },
    { key: 'strategic_priorities',  type: 'lines',    label: 'Strategic Priorities' },
    { key: 'immediate_predictions', type: 'textarea', label: 'Predictions' },
  ],
  high_risk: [
    { key: 'risk_assessment',      type: 'textarea', label: 'Risk Assessment' },
    { key: 'contributing_factors',  type: 'textarea', label: 'Contributing Factors' },
    { key: 'recommendations',      type: 'lines',    label: 'Recommendations' },
  ],
  daily: [
    { key: 'daily_assessment',     type: 'textarea', label: 'Daily Assessment' },
    { key: 'key_observations',     type: 'lines',    label: 'Key Observations' },
  ],
  final: [
    { key: 'final_assessment',     type: 'textarea', label: 'Final Assessment' },
    { key: 'lessons_learned',      type: 'lines',    label: 'Lessons Learned' },
    { key: 'recommendations',      type: 'lines',    label: 'Recommendations' },
  ],
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function riskColor(r) { return { CRITICAL:'critical', HIGH:'high', MODERATE:'medium', LOW:'low' }[r] || 'muted'; }
function confColor(c) { return c >= 0.8 ? 'text-accent-green' : c >= 0.6 ? 'text-accent-orange' : 'text-risk-critical'; }

function fmtDate(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month:'short', day:'numeric' }) + ', ' +
           d.toLocaleTimeString('en-US', { hour:'numeric', minute:'2-digit' });
  } catch { return iso.slice(0,16); }
}

function getNestedValue(obj, path) {
  return path.split('.').reduce((o, k) => o?.[k], obj);
}

// ─── Sub: Status Banner ───────────────────────────────────────────────────────
function StatusBanner({ status, loading, onRefresh }) {
  return (
    <div className="bg-surface-2 border border-border-subtle rounded-[10px] p-3 flex items-center justify-between">
      <div className="flex items-center gap-6">
        <div>
          <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">OBJ-3</div>
          <div className="text-text-primary text-xs font-semibold">AI Disaster Reporter</div>
        </div>
        {loading ? (
          <div className="flex items-center gap-2 text-text-muted"><Spinner size="sm" /><span className="text-[10px] font-mono">Connecting...</span></div>
        ) : status ? (
          <>
            <div>
              <div className="text-text-muted text-[9px] font-mono uppercase">Backend</div>
              <div className="text-text-secondary text-xs font-mono">{status.backend}</div>
            </div>
            <div>
              <div className="text-text-muted text-[9px] font-mono uppercase">Corpus</div>
              <div className="text-text-secondary text-xs font-mono">{status.corpus_chunks} chunks</div>
            </div>
            <div className="flex items-center gap-3">
              {status.reporter_loaded
                ? <span className="flex items-center gap-1 text-[10px] font-mono text-accent-green"><CheckCircle className="w-3 h-3"/>Reporter OK</span>
                : <span className="flex items-center gap-1 text-[10px] font-mono text-risk-critical"><XCircle className="w-3 h-3"/>Reporter Down</span>}
              {status.gemini && (
                status.gemini.api_key_set
                  ? <span className="flex items-center gap-1 text-[10px] font-mono text-accent-green"><CheckCircle className="w-3 h-3"/>Gemini</span>
                  : <span className="flex items-center gap-1 text-[10px] font-mono text-risk-critical"><XCircle className="w-3 h-3"/>No API Key</span>
              )}
            </div>
          </>
        ) : (
          <span className="flex items-center gap-1.5 text-[10px] font-mono text-text-muted">
            <span className="w-1.5 h-1.5 rounded-full bg-text-muted"/>API offline — live features available when backend is running
          </span>
        )}
      </div>
      <button onClick={onRefresh} className="p-1.5 text-text-muted hover:text-text-primary rounded hover:bg-surface-3 transition-colors">
        <RefreshCw className="w-3.5 h-3.5"/>
      </button>
    </div>
  );
}

// ─── Sub: Stat Cards ──────────────────────────────────────────────────────────
function StatCards({ reports }) {
  if (!reports || !reports.length) return null;
  const total = reports.length;
  const pending = reports.filter(r => r.human_review_required).length;
  const incidents = reports.filter(r => r.report_type === 'incident').length;
  const avg = total > 0 ? reports.reduce((s,r) => s + (r.confidence||0), 0) / total : 0;

  const cards = [
    { icon: FileText, label: 'Total Reports', value: total, color: '' },
    { icon: Clock,    label: 'Pending Review', value: pending, color: pending > 0 ? 'text-accent-orange' : '' },
    { icon: Shield,   label: 'Incidents',      value: incidents, color: incidents > 0 ? 'text-risk-critical' : '' },
    { icon: BarChart3,label: 'Avg Confidence',  value: (avg*100).toFixed(0)+'%', color: confColor(avg) },
  ];
  return (
    <div className="grid grid-cols-4 gap-3">
      {cards.map((c,i) => (
        <div key={i} className={`bg-surface-1 border border-border-subtle rounded-[10px] p-3 shadow-card animate-fade-up`}
          style={{ animationDelay: `${i*60}ms` }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-[8px] bg-surface-2 flex items-center justify-center flex-shrink-0">
              <c.icon className="w-4 h-4 text-text-muted"/>
            </div>
            <div>
              <div className={`font-display text-xl font-bold leading-none ${c.color || 'text-text-primary'}`}>{c.value}</div>
              <div className="text-[9px] font-semibold uppercase tracking-wide text-text-muted mt-0.5">{c.label}</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Sub: Viewer Panel ────────────────────────────────────────────────────────
function ViewerPanel({ reportId, reportMeta, onClose, onRefresh }) {
  const [tab, setTab] = useState('view');
  const [fullReport, setFullReport] = useState(null);
  const [editValues, setEditValues] = useState({});
  const [saveStatus, setSaveStatus] = useState(null);
  const [aiSummary, setAiSummary] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);

  // Fetch full report JSON for edit/summary tabs
  useEffect(() => {
    if (!reportId) return;
    setTab('view');
    setSaveStatus(null);
    setAiSummary(null);
    setEditValues({});
    fetch(apiUrl(`/api/reports/${reportId}`))
      .then(r => r.json())
      .then(setFullReport)
      .catch(() => setFullReport(null));
  }, [reportId]);

  if (!reportId) return null;

  const iframeUrl = apiUrl(`/api/reports/${reportId}/render?format=html`);
  const reportType = reportMeta?.report_type || fullReport?.report_type || 'daily';
  const editFields = EDIT_SECTIONS[reportType] || EDIT_SECTIONS.daily;

  async function saveEdits() {
    const updates = {};
    for (const [k, v] of Object.entries(editValues)) {
      if (v !== undefined && v !== '') updates[k] = v;
    }
    if (!Object.keys(updates).length) return;
    setSaveStatus('saving');
    try {
      const res = await fetch(apiUrl(`/api/reports/${reportId}`), {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      const data = await res.json();
      setSaveStatus(data.updated?.length > 0 ? 'saved' : 'no-change');
      if (data.updated?.length > 0 && onRefresh) onRefresh();
    } catch { setSaveStatus('error'); }
  }

  async function generateSummary() {
    setAiLoading(true);
    try {
      const res = await fetch(apiUrl(`/api/reports/${reportId}/summarize`), { method: 'POST' });
      const data = await res.json();
      setAiSummary(data.summary || 'No summary returned');
    } catch (e) { setAiSummary(`Error: ${e.message}`); }
    finally { setAiLoading(false); }
  }

  function buildQuickSummary() {
    if (!fullReport) return '';
    const r = fullReport;
    return [
      `Type: ${r.report_type} | Risk: ${r.risk_level} | Mode: ${r.operating_mode}`,
      `Incident: ${r.incident_id}`,
      `Generated: ${r.generated_at}`,
      `Confidence: ${r.report_confidence}`,
      r.situation_summary ? `\nSituation:\n${r.situation_summary}` : '',
      r.risk_assessment ? `\nAssessment:\n${r.risk_assessment}` : '',
      r.daily_assessment ? `\nAssessment:\n${r.daily_assessment}` : '',
    ].filter(Boolean).join('\n');
  }

  function copyText(text) { navigator.clipboard.writeText(text).catch(() => {}); }

  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="absolute inset-0 bg-black/20" style={{ backdropFilter: 'blur(3px)' }}/>
      <div className="relative w-[72%] max-w-[1000px] bg-surface-1 border-l border-border-subtle flex flex-col shadow-card-lg animate-fade-up">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle bg-surface-2 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-text-secondary">{reportId}</span>
            {reportMeta && <Badge color={riskColor(reportMeta.risk_level)}>{reportMeta.risk_level}</Badge>}
            {reportMeta && <Badge color="muted">{reportMeta.report_type}</Badge>}
          </div>
          <div className="flex items-center gap-2">
            {/* Tabs */}
            <div className="flex bg-surface-3 rounded-[7px] p-0.5 gap-0.5">
              {[['view','Report'],['edit','Edit'],['summary','Summary']].map(([id,label]) => (
                <button key={id} onClick={() => setTab(id)}
                  className={`px-2.5 py-1 rounded-[5px] text-[10px] font-mono transition-colors ${tab===id ? 'bg-surface-1 text-text-primary shadow-card font-semibold' : 'text-text-muted hover:text-text-secondary'}`}>
                  {label}
                </button>
              ))}
            </div>
            <a href={apiUrl(`/api/reports/${reportId}`)} target="_blank" rel="noreferrer"
              className="text-[9px] font-mono text-text-muted hover:text-accent-blue">JSON</a>
            <button onClick={onClose} className="text-text-muted hover:text-text-primary ml-1"><XCircle className="w-4 h-4"/></button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {tab === 'view' && (
            <iframe src={iframeUrl} className="w-full h-full border-none bg-white" title="Report"/>
          )}

          {tab === 'edit' && fullReport && (
            <div className="p-4 space-y-4">
              {/* Quality suggestions */}
              {fullReport.report_confidence < 0.7 && (
                <div className="p-3 rounded-[8px] bg-accent-orange/10 border border-accent-orange/30">
                  <div className="text-[10px] font-semibold text-accent-orange uppercase mb-1">Quality Warning</div>
                  <div className="text-[10px] text-text-secondary">Confidence {(fullReport.report_confidence*100).toFixed(0)}% is below threshold. Consider editing key sections.</div>
                </div>
              )}
              {/* Edit fields */}
              {editFields.map(field => {
                const current = getNestedValue(fullReport, field.key) || '';
                const val = editValues[field.key] ?? (Array.isArray(current) ? current.join('\n') : String(current));
                return (
                  <label key={field.key} className="block">
                    <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider">{field.label}</span>
                    {field.type === 'textarea' || field.type === 'lines' ? (
                      <textarea value={val} onChange={e => setEditValues(p => ({...p, [field.key]: field.type==='lines' ? e.target.value.split('\n') : e.target.value}))}
                        rows={field.type==='lines' ? 4 : 3}
                        className="mt-0.5 w-full px-2.5 py-2 bg-surface-2 border border-border-subtle rounded-[6px] text-[12px] text-text-primary font-mono focus:border-accent outline-none resize-y"/>
                    ) : (
                      <input type="text" value={val} onChange={e => setEditValues(p => ({...p, [field.key]: e.target.value}))}
                        className="mt-0.5 w-full px-2.5 py-2 bg-surface-2 border border-border-subtle rounded-[6px] text-[12px] text-text-primary font-mono focus:border-accent outline-none"/>
                    )}
                  </label>
                );
              })}
              <div className="flex items-center gap-3 pt-2">
                <button onClick={saveEdits}
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-[7px] bg-accent text-white text-[11px] font-semibold hover:bg-accent-hover transition-colors">
                  Save Changes
                </button>
                {saveStatus === 'saved' && <span className="text-[10px] font-mono text-accent-green">Saved</span>}
                {saveStatus === 'error' && <span className="text-[10px] font-mono text-risk-critical">Save failed</span>}
              </div>
            </div>
          )}

          {tab === 'summary' && (
            <div className="p-4 space-y-4">
              {/* Quick brief */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider">Quick Brief</span>
                  <button onClick={() => copyText(buildQuickSummary())} className="text-text-muted hover:text-text-primary"><Copy className="w-3 h-3"/></button>
                </div>
                <pre className="p-3 bg-surface-2 border border-border-subtle rounded-[8px] text-[11px] font-mono text-text-secondary whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
                  {buildQuickSummary() || 'Loading...'}
                </pre>
              </div>
              {/* AI summary */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider">AI Executive Summary</span>
                  <div className="flex gap-2">
                    {aiSummary && <button onClick={() => copyText(aiSummary)} className="text-text-muted hover:text-text-primary"><Copy className="w-3 h-3"/></button>}
                  </div>
                </div>
                {aiSummary ? (
                  <pre className="p-3 bg-surface-2 border border-accent-green/30 rounded-[8px] text-[11px] font-mono text-text-secondary whitespace-pre-wrap leading-relaxed">
                    {aiSummary}
                  </pre>
                ) : (
                  <button onClick={generateSummary} disabled={aiLoading}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-[7px] bg-accent text-white text-[11px] font-semibold hover:bg-accent-hover transition-colors disabled:opacity-40">
                    {aiLoading ? <><Spinner size="sm"/>Generating...</> : <><Zap className="w-3 h-3"/>Generate AI Summary</>}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Sub: Reports Tab ─────────────────────────────────────────────────────────
function ReportsTab({ reports, loading, onRefresh }) {
  const [typeFilter, setTypeFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [sortCol, setSortCol] = useState('generated_at');
  const [sortDir, setSortDir] = useState('desc');
  const [viewId, setViewId] = useState(null);
  const searchRef = useRef(null);

  // Keyboard shortcut: / to focus search, Esc to close viewer
  useEffect(() => {
    function onKey(e) {
      if (e.key === '/' && !['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) { e.preventDefault(); searchRef.current?.focus(); }
      if (e.key === 'Escape') setViewId(null);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Filter + sort
  let filtered = reports || [];
  if (typeFilter !== 'all') filtered = filtered.filter(r => r.report_type === typeFilter);
  if (search) {
    const q = search.toLowerCase();
    filtered = filtered.filter(r =>
      (r.incident_id||'').toLowerCase().includes(q) ||
      (r.report_type||'').toLowerCase().includes(q) ||
      (r.risk_level||'').toLowerCase().includes(q) ||
      (r.review_status||'').toLowerCase().includes(q)
    );
  }
  filtered = [...filtered].sort((a,b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (sortCol === 'confidence') { va = va||0; vb = vb||0; }
    if (va < vb) return sortDir === 'asc' ? -1 : 1;
    if (va > vb) return sortDir === 'asc' ? 1 : -1;
    return 0;
  });

  function toggleSort(col) {
    if (sortCol === col) setSortDir(d => d==='asc'?'desc':'asc');
    else { setSortCol(col); setSortDir('desc'); }
  }

  async function handleDelete(id) {
    if (!confirm(`Delete report ${id}?`)) return;
    try { await fetch(apiUrl(`/api/reports/${id}`), { method: 'DELETE' }); onRefresh(); } catch {}
  }

  const typeCounts = {};
  (reports||[]).forEach(r => { typeCounts[r.report_type] = (typeCounts[r.report_type]||0)+1; });

  const viewMeta = viewId ? (reports||[]).find(r => r.id === viewId) : null;

  return (
    <>
      <StatCards reports={reports}/>

      {/* Toolbar: type tabs + search */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-1 bg-surface-2 border border-border-subtle rounded-[7px] p-0.5">
          {REPORT_TYPES.map(t => {
            const count = t==='all' ? (reports||[]).length : (typeCounts[t]||0);
            return (
              <button key={t} onClick={() => setTypeFilter(t)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-[5px] text-[10px] font-mono transition-colors
                  ${typeFilter===t ? 'bg-surface-1 text-text-primary shadow-card font-semibold' : 'text-text-muted hover:text-text-secondary'}`}>
                {t==='all'?'All':t.replace('_',' ')}
                <span className="text-[8px] bg-surface-3 px-1 rounded">{count}</span>
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 bg-surface-2 border border-border-subtle rounded-[7px] px-2.5 py-1 w-56 focus-within:border-accent transition-colors">
            <Search className="w-3 h-3 text-text-muted flex-shrink-0"/>
            <input ref={searchRef} value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search reports... ( / )" className="bg-transparent outline-none text-[11px] font-mono text-text-primary w-full placeholder:text-text-muted"/>
          </div>
          <button onClick={onRefresh} className="p-1.5 text-text-muted hover:text-text-primary rounded hover:bg-surface-2 transition-colors" title="Refresh">
            <RefreshCw className="w-3.5 h-3.5"/>
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="bg-surface-1 border border-border-subtle rounded-[10px] shadow-card overflow-hidden">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b-2 border-border-subtle bg-surface-2">
              {[
                { key:'generated_at', label:'Date / Time' },
                { key:'report_type',  label:'Type' },
                { key:'risk_level',   label:'Risk' },
                { key:null,           label:'Incident ID' },
                { key:'confidence',   label:'Confidence' },
                { key:null,           label:'Review' },
                { key:null,           label:'' },
              ].map((col,i) => (
                <th key={i} className={`text-left text-text-muted font-mono py-2.5 px-3 uppercase text-[9px] tracking-wider ${col.key ? 'cursor-pointer hover:text-accent' : ''}`}
                  onClick={() => col.key && toggleSort(col.key)}>
                  <span className="flex items-center gap-1">{col.label}
                    {col.key === sortCol && <ArrowUpDown className="w-2.5 h-2.5"/>}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} className="py-8 text-center text-text-muted"><Spinner size="sm" className="inline-block mr-2"/>Loading...</td></tr>
            ) : filtered.length === 0 ? (
              <tr><td colSpan={7} className="py-12 text-center">
                <FileText className="w-8 h-8 text-text-muted opacity-30 mx-auto mb-2"/>
                <div className="text-[11px] font-mono text-text-muted">
                  {reports === null ? 'Reports will appear here when the backend API is running' : 'No reports generated yet — use the Generate tab to create one'}
                </div>
              </td></tr>
            ) : filtered.map((r,i) => (
              <tr key={r.id} className="border-b border-border-subtle/50 hover:bg-accent/[0.03] transition-colors cursor-pointer animate-fade-up"
                style={{ animationDelay:`${i*30}ms` }}
                onClick={() => setViewId(r.id)}>
                <td className="py-2.5 px-3 font-mono text-text-secondary">{fmtDate(r.generated_at)}</td>
                <td className="py-2.5 px-3"><Badge color="muted">{r.report_type}</Badge></td>
                <td className="py-2.5 px-3"><Badge color={riskColor(r.risk_level)}>{r.risk_level}</Badge></td>
                <td className="py-2.5 px-3 font-mono text-text-secondary truncate max-w-[140px]">{r.incident_id}</td>
                <td className="py-2.5 px-3">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-surface-3 rounded-full overflow-hidden">
                      <div className="h-full rounded-full bg-accent-green" style={{ width:`${(r.confidence||0)*100}%` }}/>
                    </div>
                    <span className={`font-mono ${confColor(r.confidence||0)}`}>{((r.confidence||0)*100).toFixed(0)}%</span>
                  </div>
                </td>
                <td className="py-2.5 px-3">
                  <Badge color={r.human_review_required ? 'orange' : 'green'}>
                    {r.review_status || (r.human_review_required ? 'PENDING' : 'OK')}
                  </Badge>
                </td>
                <td className="py-2.5 px-3" onClick={e => e.stopPropagation()}>
                  <button onClick={() => handleDelete(r.id)} className="text-text-muted hover:text-risk-critical p-1 rounded transition-colors" title="Delete">
                    <Trash2 className="w-3 h-3"/>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Viewer panel */}
      {viewId && <ViewerPanel reportId={viewId} reportMeta={viewMeta} onClose={() => setViewId(null)} onRefresh={onRefresh}/>}
    </>
  );
}

// ─── Sub: Generate Tab ────────────────────────────────────────────────────────
function GenerateTab({ onGenerated }) {
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [rerunMode, setRerunMode] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileInputRef = useRef(null);

  const [form, setForm] = useState({
    risk_level: 'HIGH', firms_hotspot_count: 0, temperature_max: '', wind_speed_mph: '',
    relative_humidity: '', soil_moisture: '', propagator_summary: '', xgboost_cells_json: '',
    obj2_simulation_json: '', operator_notes: '', report_type_override: 'auto', backend_override: '',
    // Rerun fields
    grid_id: '', region: 'california', rerun_temperature_f: '', rerun_wind: '', rerun_rh: '',
    rerun_soil: '', rerun_fwi: '',
  });

  function set(k,v) { setForm(p => ({...p,[k]:v})); }
  function applyPreset(key) {
    if (key === 'clear') { setForm(p => ({...p, risk_level:'HIGH', firms_hotspot_count:0, temperature_max:'', wind_speed_mph:'', relative_humidity:'', soil_moisture:''})); return; }
    const pr = PRESETS[key];
    setForm(p => ({...p, ...pr}));
  }

  function handleDrop(e) { e.preventDefault(); addFiles(e.dataTransfer.files); }
  function addFiles(fl) { setUploadedFiles(prev => [...prev, ...Array.from(fl)]); }
  function removeFile(i) { setUploadedFiles(prev => prev.filter((_,j)=>j!==i)); }

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitting(true); setResult(null);
    try {
      if (rerunMode) {
        if (!form.grid_id.trim()) { setResult({success:false, error:'Grid ID is required'}); setSubmitting(false); return; }
        const body = new FormData();
        body.append('grid_id', form.grid_id);
        body.append('region', form.region);
        if (form.rerun_temperature_f) body.append('temperature_f', form.rerun_temperature_f);
        if (form.rerun_wind) body.append('wind_speed_mph', form.rerun_wind);
        if (form.rerun_rh) body.append('relative_humidity', form.rerun_rh);
        if (form.rerun_soil) body.append('soil_moisture', form.rerun_soil);
        if (form.rerun_fwi) body.append('fire_weather_index', form.rerun_fwi);
        if (form.operator_notes) body.append('operator_notes', form.operator_notes);
        if (form.backend_override) body.append('backend_override', form.backend_override);
        const res = await fetch(apiUrl('/api/rerun'), { method:'POST', body });
        setResult(await res.json());
      } else {
        const body = new FormData();
        ['risk_level','firms_hotspot_count','temperature_max','wind_speed_mph','relative_humidity','soil_moisture',
         'propagator_summary','xgboost_cells_json','obj2_simulation_json','operator_notes','report_type_override','backend_override'
        ].forEach(k => { if(form[k] !== '' && form[k] !== undefined) body.append(k, form[k]); });
        uploadedFiles.forEach(f => body.append('files', f));
        const res = await fetch(apiUrl('/api/generate'), { method:'POST', body });
        setResult(await res.json());
      }
      if (onGenerated) onGenerated();
    } catch(err) {
      const msg = err.message?.includes('fetch') || err.message?.includes('network')
        ? 'Cannot reach API server. Start the backend first: cd model-pipeline && python scripts/run_dashboard.py'
        : err.message;
      setResult({success:false, error:msg});
    }
    finally { setSubmitting(false); }
  }

  const INPUT = "w-full px-2.5 py-2 bg-surface-2 border border-border-subtle rounded-[6px] text-[12px] text-text-primary font-mono focus:border-accent outline-none";
  const LABEL = "text-[9px] font-mono text-text-muted uppercase tracking-wider block mb-0.5";

  return (
    <form onSubmit={handleSubmit} className="grid grid-cols-[1fr_280px] gap-5">
      {/* ── Left column ── */}
      <div className="space-y-4">
        {/* Presets */}
        <Card>
          <div className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">Quick Fill</div>
          <div className="flex gap-2">
            {[['low','Quiet / Low','green'],['high','Active / High','orange'],['emergency','Emergency / Critical','critical'],['clear','Clear','muted']].map(([k,l,c])=>(
              <button key={k} type="button" onClick={() => applyPreset(k)}
                className={`flex-1 py-1.5 rounded-[7px] border text-[10px] font-mono font-semibold transition-colors
                  border-${c==='muted'?'border-subtle':`risk-${c}/40`} text-${c==='muted'?'text-secondary':`risk-${c}`}
                  hover:bg-${c==='muted'?'surface-2':`risk-${c}/10`}`}>
                {l}
              </button>
            ))}
          </div>
        </Card>

        {/* Risk level */}
        <Card>
          <div className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">Risk Level</div>
          <div className="grid grid-cols-4 gap-2">
            {RISK_LEVELS.map(r => (
              <label key={r} className={`flex flex-col items-center gap-1 py-2.5 rounded-[7px] border-2 cursor-pointer transition-colors
                ${form.risk_level===r ? `border-risk-${riskColor(r)} bg-risk-${riskColor(r)}/10` : 'border-border-subtle hover:border-border-default'}`}>
                <input type="radio" name="risk" value={r} checked={form.risk_level===r} onChange={() => set('risk_level',r)} className="sr-only"/>
                <div className={`w-2.5 h-2.5 rounded-full bg-risk-${riskColor(r)}`}/>
                <span className="text-[10px] font-mono font-bold text-text-primary">{r}</span>
              </label>
            ))}
          </div>
        </Card>

        {/* Environmental */}
        <Card>
          <div className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">Environmental Conditions</div>
          <div className="grid grid-cols-3 gap-3">
            {[['temperature_max','Temp Max (°F)','number'],['wind_speed_mph','Wind (mph)','number'],['relative_humidity','RH (%)','number'],
              ['soil_moisture','Soil Moisture','number'],['firms_hotspot_count','FIRMS Hotspots','number']].map(([k,l,t])=>(
              <label key={k} className="block"><span className={LABEL}>{l}</span>
                <input type={t} step="any" value={form[k]} onChange={e => set(k, t==='number' ? (e.target.value===''?'':Number(e.target.value)) : e.target.value)}
                  placeholder={k==='firms_hotspot_count'?'0':'opt'} className={INPUT}/></label>
            ))}
          </div>
        </Card>

        {/* ML Outputs */}
        <Card>
          <div className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">ML Outputs (optional)</div>
          <div className="space-y-3">
            <label className="block"><span className={LABEL}>Propagator / Spread Summary</span>
              <textarea value={form.propagator_summary} onChange={e=>set('propagator_summary',e.target.value)} rows={2} placeholder="Paste spread model output..." className={INPUT + ' resize-y'}/></label>
            <label className="block"><span className={LABEL}>XGBoost Top Cells (JSON array)</span>
              <textarea value={form.xgboost_cells_json} onChange={e=>set('xgboost_cells_json',e.target.value)} rows={2} placeholder='[{"grid_id":"...","score":0.8}]' className={INPUT + ' resize-y'}/></label>
            <label className="block"><span className={LABEL}>OBJ-2 Simulation (JSON)</span>
              <textarea value={form.obj2_simulation_json} onChange={e=>set('obj2_simulation_json',e.target.value)} rows={2} placeholder='{"spread_area_km2":...}' className={INPUT + ' resize-y'}/></label>
          </div>
        </Card>

        {/* Operator Notes + Files */}
        <Card>
          <div className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">Operator Input</div>
          <label className="block mb-3"><span className={LABEL}>Notes</span>
            <textarea value={form.operator_notes} onChange={e=>set('operator_notes',e.target.value)} rows={3} placeholder="Field observations, local conditions..." className={INPUT + ' resize-y'}/></label>
          <div className={LABEL}>File Uploads</div>
          <div onDragOver={e=>e.preventDefault()} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-border-default rounded-[7px] p-4 text-center cursor-pointer hover:border-accent hover:bg-accent/[0.03] transition-colors">
            <Upload className="w-5 h-5 text-text-muted mx-auto mb-1"/>
            <div className="text-[10px] text-text-muted font-mono">Drop files or click to browse</div>
            <input ref={fileInputRef} type="file" multiple className="hidden" onChange={e => addFiles(e.target.files)}/>
          </div>
          {uploadedFiles.length > 0 && (
            <div className="mt-2 space-y-1">
              {uploadedFiles.map((f,i) => (
                <div key={i} className="flex items-center gap-2 bg-surface-2 border border-border-subtle rounded-[5px] px-2 py-1">
                  <span className="text-[9px] font-mono text-text-secondary flex-1 truncate">{f.name}</span>
                  <span className="text-[9px] font-mono text-text-muted">{(f.size/1024).toFixed(1)}KB</span>
                  <button type="button" onClick={() => removeFile(i)} className="text-text-muted hover:text-risk-critical"><XCircle className="w-3 h-3"/></button>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* ── Right rail ── */}
      <div className="space-y-4">
        <div className="sticky top-0 space-y-4">
          {/* Settings */}
          <Card>
            <div className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">Settings</div>
            <label className="block mb-2"><span className={LABEL}>Report Type</span>
              <select value={form.report_type_override} onChange={e=>set('report_type_override',e.target.value)} className={INPUT}>
                <option value="auto">Auto-detect</option>
                <option value="daily">Daily</option><option value="high_risk">High Risk</option>
                <option value="incident">Incident</option><option value="final">Final</option>
              </select></label>
            <label className="block"><span className={LABEL}>LLM Backend</span>
              <select value={form.backend_override} onChange={e=>set('backend_override',e.target.value)} className={INPUT}>
                <option value="">Config Default</option>
                <option value="ollama">Ollama (local)</option><option value="gemini_dev">Gemini Dev</option>
              </select></label>
          </Card>

          {/* Re-run toggle */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider">Re-run with Local Data</span>
              <button type="button" onClick={() => setRerunMode(v=>!v)}
                className={`w-8 h-4 rounded-full transition-colors flex items-center ${rerunMode?'bg-accent justify-end':'bg-surface-3 justify-start'}`}>
                <div className="w-3 h-3 rounded-full bg-white shadow mx-0.5"/>
              </button>
            </div>
            {rerunMode && (
              <div className="space-y-2 mt-2">
                <label className="block"><span className={LABEL}>Grid ID *</span>
                  <input type="text" value={form.grid_id} onChange={e=>set('grid_id',e.target.value)} required placeholder="82287bffffffffff" className={INPUT}/></label>
                <label className="block"><span className={LABEL}>Region</span>
                  <select value={form.region} onChange={e=>set('region',e.target.value)} className={INPUT}>
                    <option value="california">California</option><option value="texas">Texas</option>
                  </select></label>
                <div className="grid grid-cols-2 gap-2">
                  <label className="block"><span className={LABEL}>Temp (°F)</span>
                    <input type="number" step="any" value={form.rerun_temperature_f} onChange={e=>set('rerun_temperature_f',e.target.value)} className={INPUT}/></label>
                  <label className="block"><span className={LABEL}>Wind (mph)</span>
                    <input type="number" step="any" value={form.rerun_wind} onChange={e=>set('rerun_wind',e.target.value)} className={INPUT}/></label>
                  <label className="block"><span className={LABEL}>RH (%)</span>
                    <input type="number" step="any" value={form.rerun_rh} onChange={e=>set('rerun_rh',e.target.value)} className={INPUT}/></label>
                  <label className="block"><span className={LABEL}>FWI</span>
                    <input type="number" step="any" value={form.rerun_fwi} onChange={e=>set('rerun_fwi',e.target.value)} className={INPUT}/></label>
                </div>
              </div>
            )}
          </Card>

          {/* Submit */}
          <button type="submit" disabled={submitting || (rerunMode && !form.grid_id.trim())}
            className="w-full flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-[7px] bg-accent text-white text-[12px] font-semibold hover:bg-accent-hover transition-colors disabled:opacity-40">
            {submitting ? <><Spinner size="sm"/>Generating...</> : <><Send className="w-3.5 h-3.5"/>{rerunMode ? 'Re-run & Generate' : 'Generate Report'}</>}
          </button>

          {/* Result */}
          {result && (
            <div className={`p-3 rounded-[8px] border text-[10px] font-mono ${result.success
              ? 'bg-accent-green/10 border-accent-green/40 text-accent-green'
              : 'bg-risk-critical/10 border-risk-critical/40 text-risk-critical'}`}>
              {result.success ? (
                <div className="space-y-0.5">
                  <div className="font-semibold">Report generated</div>
                  <div>Type: {result.report_type} — {result.incident_id || result.grid_id}</div>
                  {result.confidence != null && <div>Confidence: {(result.confidence*100).toFixed(0)}%</div>}
                  {result.latency_ms != null && <div>Latency: {result.latency_ms.toFixed(0)}ms</div>}
                  {result.human_review_required && <div className="text-accent-orange">Human review required</div>}
                </div>
              ) : <div>Error: {result.error}</div>}
            </div>
          )}
        </div>
      </div>
    </form>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function OBJ3Reporter() {
  const [tab, setTab] = useState('reports');
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const { data: status, loading: statusLoading, refresh: refreshStatus } = useAPI('/api/status', { interval: 30000 });
  const { data: reports, loading: reportsLoading, refresh: refreshReports } = useAPI('/api/reports?limit=500');

  function handleGenerated() { refreshReports(); refreshStatus(); }

  async function triggerPipelineReport() {
    setPipelineRunning(true);
    try {
      const res = await fetch(apiUrl('/api/generate-from-pipeline'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ regions: ['california', 'texas'] }),
      });
      const data = await res.json();
      if (!res.ok) { alert('Pipeline report failed: ' + (data.detail || res.statusText)); return; }
      const ok = (data.reports || []).filter(r => r.success).length;
      const fail = (data.reports || []).filter(r => !r.success).length;
      alert(`Pipeline reports: ${ok} generated, ${fail} failed.`);
      refreshReports(); refreshStatus();
    } catch (e) { alert('Request failed: ' + e.message); }
    finally { setPipelineRunning(false); }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 space-y-5 max-w-[1400px] mx-auto">
        <StatusBanner status={status} loading={statusLoading} onRefresh={refreshStatus}/>

        {/* Tab bar */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 bg-surface-2 border border-border-subtle rounded-[7px] p-0.5 w-fit">
            {[['reports','Reports',FileText],['generate','Generate',Send],['config','Config',MessageSquare]].map(([id,label,Icon])=>(
              <button key={id} onClick={() => setTab(id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-[5px] text-[11px] font-mono transition-colors
                  ${tab===id ? 'bg-surface-1 text-text-primary shadow-card font-semibold' : 'text-text-muted hover:text-text-secondary'}`}>
                <Icon className="w-3 h-3"/>{label}
              </button>
            ))}
          </div>
          <button onClick={triggerPipelineReport} disabled={pipelineRunning}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-[7px] text-[11px] font-mono font-semibold
              bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            {pipelineRunning ? <><Loader2 className="w-3 h-3 animate-spin"/>Running...</> : <><Zap className="w-3 h-3"/>Run Pipeline Report</>}
          </button>
        </div>

        {/* Tab content */}
        {tab === 'reports' && <ReportsTab reports={reports} loading={reportsLoading} onRefresh={refreshReports}/>}

        {tab === 'generate' && <GenerateTab onGenerated={handleGenerated}/>}

        {tab === 'config' && (
          <div className="space-y-5">
            {/* State Machine */}
            <Section title="State Machine — Mode Resolution" icon={MessageSquare}
              action={<span className="text-[9px] font-mono text-text-muted">state_machine.py</span>}>
              <div className="overflow-x-auto">
                <table className="w-full text-[11px]">
                  <thead><tr className="border-b border-border-subtle">
                    {['risk_level','firms_count','is_deployable','→ mode','disagreement'].map(c=>(
                      <th key={c} className="text-left text-text-muted font-mono py-1.5 pr-4 uppercase text-[9px] tracking-wider">{c}</th>))}
                  </tr></thead>
                  <tbody>{MODE_MATRIX.map((r,i)=>(
                    <tr key={i} className="border-b border-border-subtle/50 hover:bg-surface-2/50 transition-colors">
                      <td className="py-1.5 pr-4 font-mono text-text-secondary">{r.risk}</td>
                      <td className="py-1.5 pr-4 font-mono text-text-secondary">{r.hotspots}</td>
                      <td className="py-1.5 pr-4 font-mono text-text-secondary">{String(r.deployable)}</td>
                      <td className="py-1.5 pr-4"><Badge color={{ QUIET:'green',ACTIVE:'orange',EMERGENCY:'critical' }[r.mode]}>{r.mode}</Badge></td>
                      <td className={`py-1.5 pr-4 font-mono text-[10px] ${r.disagreement?'text-accent-orange':'text-text-muted'}`}>{r.disagreement?'YES':'false'}</td>
                    </tr>))}</tbody>
                </table>
              </div>
            </Section>

            {/* Watchdog */}
            <Section title="Watchdog Configuration" icon={AlertTriangle}
              action={<span className="text-[9px] font-mono text-text-muted">schema_config.yaml</span>}>
              <div className="grid grid-cols-3 gap-3">
                {Object.entries(WATCHDOG_CONFIG.modes).map(([name,cfg])=>(
                  <div key={name} className={`bg-surface-2 border rounded-[7px] p-3 ${name===WATCHDOG_CONFIG.current_mode?'border-accent-green/40':'border-border-subtle'}`}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-text-primary text-xs font-mono font-semibold uppercase">{name}</span>
                      {name===WATCHDOG_CONFIG.current_mode && <Badge color="green">ACTIVE</Badge>}
                    </div>
                    <div className="space-y-1 text-[10px] font-mono">
                      <div className="flex justify-between"><span className="text-text-muted">poll</span><span className="text-text-secondary">{cfg.poll_interval_min}min</span></div>
                      <div className="flex justify-between"><span className="text-text-muted">pipeline</span><span className="text-text-secondary">{cfg.pipeline_interval_hr}hr</span></div>
                      <div className="flex justify-between"><span className="text-text-muted">resolution</span><span className="text-text-secondary">{cfg.resolution_km}km</span></div>
                    </div>
                  </div>))}
              </div>
            </Section>
          </div>
        )}
      </div>
    </div>
  );
}
