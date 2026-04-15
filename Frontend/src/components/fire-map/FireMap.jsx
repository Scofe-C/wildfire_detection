import { useState, useMemo, useEffect } from 'react';
import {
  Flame, Wind, Thermometer, Droplets, Activity, TreePine,
  FileText, ChevronRight, Info, Crosshair, MapPin, Mountain,
  Pencil, Save, Download, X, EyeOff, Navigation, Layers, BarChart3, QrCode,
} from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Polygon, Polyline, CircleMarker, Tooltip, Marker, useMap } from 'react-leaflet';
import L from 'leaflet';
import { useTheme } from '../ui/ThemeProvider';
import { CALIFORNIA_CELLS_ENRICHED, TEXAS_CELLS_ENRICHED, getRiskTier } from '../../data/mockGridData';
import { OBJ2_SPREAD } from '../../data/mockMapData';
import { apiUrl, normalizeCell, fmt } from '../../api';
import {
  hexBoundary, compassLabel,
  riskColor, moistureColor, intensityColor, windSpeedOpacity,
  deriveCrownFire, TIER_COLORS,
  generateFireZoomCells,
} from './mapHelpers';

// ─── Config ───────────────────────────────────────────────────────────────────
const TILES = {
  dark:      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  light:     'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
  satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
};
const TILE_ATTRS = {
  dark:      '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
  light:     '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
  satellite: '&copy; <a href="https://www.esri.com/">Esri</a> &mdash; World Imagery',
};
const DEFAULT_CENTER = [33.5, -108];
const DEFAULT_ZOOM = 5;

function hexFill(cell, layer) {
  if (layer === 'moisture')  return moistureColor(cell.relative_humidity_2m);
  if (layer === 'intensity') return intensityColor(cell.mean_frp || 0);
  return riskColor(cell.fire_risk_score);
}
function hexRadius(cell) { return cell._isSubcell ? 10 : 32; }

// ─── Small UI ─────────────────────────────────────────────────────────────────
function RiskBadge({ tier }) {
  const c = { CRITICAL:'bg-risk-critical/20 text-risk-critical border-risk-critical/40', HIGH:'bg-risk-high/20 text-risk-high border-risk-high/40', MEDIUM:'bg-risk-medium/20 text-risk-medium border-risk-medium/40', LOW:'bg-risk-low/20 text-risk-low border-risk-low/40' }[tier];
  return <span className={`text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded border leading-none ${c}`}>{tier}</span>;
}
function Row({ icon: I, label, value, unit }) {
  const display = value == null ? '—' : typeof value === 'number' ? fmt(value) : value;
  return <div className="flex items-center justify-between py-[3px]"><span className="flex items-center gap-1.5 text-text-muted"><I className="w-3 h-3"/><span className="text-[10px] font-mono">{label}</span></span><span className="text-[10px] font-mono text-text-primary">{display}{unit && display !== '—' ? ` ${unit}` : ''}</span></div>;
}

// ─── Fly to selected cell ─────────────────────────────────────────────────────
function FlyTo({ lat, lon }) {
  const map = useMap();
  useEffect(() => {
    if (lat != null && lon != null) map.flyTo([lat, lon], Math.max(map.getZoom(), 7), { duration: 0.6 });
  }, [lat, lon, map]);
  return null;
}

// ─── Wind arrow icon factory ──────────────────────────────────────────────────
function makeWindIcon(dir, speed, dark = true) {
  const rot = (dir + 180) % 360;
  const op = windSpeedOpacity(speed);
  const color = dark ? '#93c5fd' : '#1e3a5f';
  const particleColor = dark ? '#60a5fa' : '#2563eb';
  // Animation speed: faster wind = faster particle (1.5s at 20km/h, 3s at 0)
  const dur = Math.max(0.8, 3 - (speed || 0) / 10).toFixed(1);
  return L.divIcon({
    html: `<div style="transform:rotate(${rot}deg);opacity:${op};width:26px;height:26px">
      <svg viewBox="0 0 26 26" width="26" height="26">
        <line x1="13" y1="22" x2="13" y2="6" stroke="${color}" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
        <polygon points="13,3 9.5,9 16.5,9" fill="${color}" opacity="0.7"/>
        <circle r="2.5" fill="${particleColor}">
          <animateMotion dur="${dur}s" repeatCount="indefinite" path="M13,22 L13,6" />
        </circle>
      </svg></div>`,
    className: '',
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
}

// ─── Crown badge icon factory ─────────────────────────────────────────────────
function makeCrownIcon(type) {
  const color = type === 'active' ? '#ef4444' : '#f59e0b';
  return L.divIcon({
    html: `<div style="background:#111;border:1.5px solid ${color};border-radius:50%;width:16px;height:16px;display:flex;align-items:center;justify-content:center;color:${color};font-size:9px;font-weight:bold;font-family:monospace">${type === 'active' ? 'A' : 'P'}</div>`,
    className: '',
    iconSize: [16, 16],
    iconAnchor: [-4, 18],
  });
}

// ─── Discover LAN IP via WebRTC ───────────────────────────────────────────────
async function getLanIP() {
  return new Promise((resolve) => {
    try {
      const pc = new RTCPeerConnection({ iceServers: [] });
      pc.createDataChannel('');
      pc.createOffer().then(o => pc.setLocalDescription(o)).catch(() => resolve(null));
      pc.onicecandidate = ({ candidate }) => {
        if (!candidate) return;
        const m = /([0-9]{1,3}(?:\.[0-9]{1,3}){3})/.exec(candidate.candidate);
        if (m && !m[1].startsWith('127.') && !m[1].startsWith('169.254.')) {
          resolve(m[1]);
          pc.close();
        }
      };
      setTimeout(() => resolve(null), 2000);
    } catch {
      resolve(null);
    }
  });
}

// ─── QR Modal ─────────────────────────────────────────────────────────────────
function QRModal({ onClose }) {
  const [lanIP, setLanIP] = useState(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    getLanIP().then(ip => {
      setLanIP(ip);
      setLoading(false);
    });
  }, []);

  const port = window.location.port || '80';
  const viewerUrl = lanIP
    ? `http://${lanIP}:${port}/?mode=viewer`
    : `${window.location.origin}/?mode=viewer`;

  const handleCopy = () => {
    navigator.clipboard?.writeText(viewerUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative bg-surface-1 border border-border-subtle rounded-2xl shadow-2xl p-6 flex flex-col items-center gap-4 w-72"
        onClick={e => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-3 right-3 text-text-muted hover:text-text-primary transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2">
          <QrCode className="w-4 h-4 text-accent-blue" />
          <span className="text-[11px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Mobile Viewer</span>
        </div>

        <div className="p-3 bg-white rounded-xl min-h-[216px] flex items-center justify-center">
          {loading ? (
            <div className="w-48 h-48 flex flex-col items-center justify-center gap-2">
              <div className="w-5 h-5 border-2 border-accent-blue/40 border-t-accent-blue rounded-full animate-spin" />
              <span className="text-[9px] font-mono text-gray-400">Detecting LAN IP…</span>
            </div>
          ) : (
            <QRCodeSVG value={viewerUrl} size={192} bgColor="#ffffff" fgColor="#0f172a" level="M" />
          )}
        </div>

        <div className="text-center space-y-1 w-full">
          <p className="text-[10px] font-mono text-text-muted">
            {lanIP ? `LAN IP detected · same Wi-Fi required` : `Could not detect LAN IP · using origin`}
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-2 bg-surface-2 border border-border-subtle rounded-lg w-full">
          <span className="text-[9px] font-mono text-text-muted flex-1 truncate">{viewerUrl}</span>
          <button
            onClick={handleCopy}
            className={`text-[9px] font-mono flex-shrink-0 transition-colors ${copied ? 'text-accent-green' : 'text-accent-blue hover:underline'}`}
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Layer bar ────────────────────────────────────────────────────────────────
function LayerBar({ layers, setLayers, stats, resolution, setResolution, mapStyle, setMapStyle, onQR }) {
  const setColor = k => setLayers(l => ({ ...l, colorLayer: k }));
  const toggle = k => setLayers(l => ({ ...l, [k]: !l[k] }));
  return (
    <div className="absolute top-3 left-3 right-3 md:right-[290px] z-[1000] flex flex-wrap items-center justify-between gap-1.5 px-2.5 py-2 bg-surface-1/90 backdrop-blur-md border border-border-subtle rounded-xl shadow-xl">
      <div className="flex flex-wrap items-center gap-1.5">
        <div className="flex flex-shrink-0 bg-surface-2 border border-border-subtle rounded-lg p-0.5 gap-px">
          {[{k:'risk',I:Activity,l:'Risk'},{k:'moisture',I:Droplets,l:'Moisture'},{k:'intensity',I:BarChart3,l:'Intensity'}].map(({k,I,l})=>(
            <button key={k} onClick={()=>setColor(k)} className={`flex flex-shrink-0 items-center gap-1.5 px-1.5 md:px-2.5 py-1 rounded-md text-[10px] font-mono whitespace-nowrap transition-colors ${layers.colorLayer===k?'bg-surface-0 text-text-primary font-semibold shadow-sm':'text-text-muted hover:text-text-secondary'}`}><I className="w-3 h-3"/>{l}</button>
          ))}
        </div>
        <div className="flex flex-shrink-0 items-center gap-1 md:pl-3 md:border-l border-border-subtle">
          {[{k:'wind',I:Wind,l:'Wind'},{k:'crown',I:TreePine,l:'Crown'},{k:'spread',I:Navigation,l:'Spread'}].map(({k,I,l})=>(
            <button key={k} onClick={()=>toggle(k)} className={`flex flex-shrink-0 items-center gap-1 px-1.5 md:px-2 py-1 rounded-md text-[10px] font-mono whitespace-nowrap border transition-colors ${layers[k]?'bg-accent-blue/10 text-accent-blue border-accent-blue/30':'text-text-muted border-transparent hover:text-text-secondary'}`}>
              {layers[k]?<I className="w-3 h-3"/>:<EyeOff className="w-3 h-3 opacity-40"/>}{l}
            </button>
          ))}
        </div>
        <div className="flex flex-shrink-0 items-center gap-1.5 md:pl-3 md:border-l border-border-subtle">
          {stats.critical>0&&<span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/10 text-risk-critical border-risk-critical/30 animate-pulse whitespace-nowrap">{stats.critical} CRIT</span>}
          {stats.high>0&&<span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-high/10 text-risk-high border-risk-high/30 whitespace-nowrap">{stats.high} HIGH</span>}
          {stats.activeFires>0&&<span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/10 text-risk-critical border-risk-critical/30 whitespace-nowrap"><Flame className="w-2.5 h-2.5 inline mr-0.5"/>{stats.activeFires}</span>}
        </div>
      </div>
      <div className="flex flex-shrink-0 items-center gap-1.5">
        <div className="flex bg-surface-2 border border-border-subtle rounded-lg p-0.5 gap-px">
          {[{k:'auto',l:'Auto'},{k:'dark',l:'Dark'},{k:'light',l:'Light'},{k:'satellite',l:'Sat'}].map(({k,l})=>(
            <button key={k} onClick={()=>setMapStyle(k)} className={`px-1.5 md:px-2 py-0.5 rounded-md text-[9px] font-mono whitespace-nowrap transition-colors ${mapStyle===k?'bg-surface-0 text-text-primary font-semibold':'text-text-muted hover:text-text-secondary'}`}>{l}</button>
          ))}
        </div>
        <div className="flex bg-surface-2 border border-border-subtle rounded-lg p-0.5 gap-px">
          <span className="px-1.5 md:px-2 py-0.5 rounded-md text-[9px] font-mono bg-surface-0 text-text-primary font-semibold">64km</span>
        </div>
        <button
          onClick={onQR}
          title="Open mobile viewer QR code"
          className="flex items-center gap-1 px-1.5 md:px-2 py-1 rounded-lg border border-border-subtle bg-surface-2 text-text-muted hover:text-accent-blue hover:border-accent-blue/40 hover:bg-accent-blue/8 transition-colors"
        >
          <QrCode className="w-3 h-3" />
          <span className="text-[9px] font-mono hidden md:inline">QR</span>
        </button>
      </div>
    </div>
  );
}

// ─── Legend ────────────────────────────────────────────────────────────────────
function Legend({ layers }) {
  return (
    <div className="absolute bottom-6 left-3 z-[1000] flex items-center gap-4 px-3 py-1.5 bg-surface-1/90 backdrop-blur-md border border-border-subtle rounded-xl shadow-xl">
      {layers.colorLayer==='risk'&&<div className="flex items-center gap-2">{['CRITICAL','HIGH','MEDIUM','LOW'].map(t=><div key={t} className="flex items-center gap-1"><div className="w-2.5 h-2.5 rounded-sm" style={{background:TIER_COLORS[t].fill}}/><span className="text-[9px] font-mono text-text-muted">{t}</span></div>)}</div>}
      {layers.colorLayer==='moisture'&&<div className="flex items-center gap-1"><div className="w-16 h-2 rounded" style={{background:'linear-gradient(90deg,#78350f,#d97706,#a3e635,#047857)'}}/><span className="text-[9px] font-mono text-text-muted">Dry→Wet</span></div>}
      {layers.colorLayer==='intensity'&&<div className="flex items-center gap-1"><div className="w-16 h-2 rounded" style={{background:'linear-gradient(90deg,#422006,#fbbf24,#ef4444,#7c3aed)'}}/><span className="text-[9px] font-mono text-text-muted">FRP</span></div>}
      <div className="flex items-center gap-1"><div className="w-2.5 h-2.5 rounded-full border-2 border-red-500 animate-pulse"/><span className="text-[9px] font-mono text-text-muted">Fire</span></div>
      {layers.wind&&<div className="flex items-center gap-1"><Wind className="w-3 h-3 text-sky-100/70"/><span className="text-[9px] font-mono text-text-muted">Wind</span></div>}
      {layers.spread&&<div className="flex items-center gap-1"><div className="w-5 h-1" style={{borderBottom:'2px dashed #fb923c'}}/><span className="text-[9px] font-mono text-text-muted">Spread</span></div>}
    </div>
  );
}

// ─── Edit form ────────────────────────────────────────────────────────────────
function EditForm({ cell, vals, onChange, onSave, onCancel }) {
  const fields = [
    {k:'temperature_2m',l:'Temp',u:'°C',s:.1},{k:'relative_humidity_2m',l:'RH',u:'%',s:.1,mn:0,mx:100},
    {k:'wind_speed_10m',l:'Wind Spd',u:'m/s',s:.1,mn:0},{k:'wind_direction_10m',l:'Wind Dir',u:'°',s:1,mn:0,mx:360},
    {k:'precipitation',l:'Precip',u:'mm',s:.1,mn:0},{k:'soil_moisture_0_to_7cm',l:'Soil Moist',u:'m³/m³',s:.01,mn:0,mx:.5},
    {k:'active_fire_count',l:'Fires',u:'',s:1,mn:0},
  ];
  return (
    <div className="px-4 py-3 border-b border-border-subtle bg-accent-blue/5">
      <div className="flex items-center justify-between mb-2"><span className="text-[10px] font-mono font-semibold text-accent-blue uppercase tracking-wider">Manual Override</span><button onClick={onCancel} className="text-text-muted hover:text-text-primary"><X className="w-3 h-3"/></button></div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        {fields.map(f=><label key={f.k} className="flex flex-col gap-0.5"><span className="text-[9px] font-mono text-text-muted">{f.l} <span className="opacity-40">{f.u}</span></span><input type="number" step={f.s} min={f.mn} max={f.mx} value={vals[f.k]??cell[f.k]??''} onChange={e=>onChange(f.k,e.target.value===''?undefined:+e.target.value)} className="w-full bg-surface-2 border border-border-subtle rounded px-1.5 py-1 text-[10px] font-mono text-text-primary focus:border-accent-blue/40 focus:outline-none"/></label>)}
      </div>
      <label className="flex flex-col gap-0.5 mt-2"><span className="text-[9px] font-mono text-text-muted">Notes</span><textarea value={vals.notes??cell.notes??''} onChange={e=>onChange('notes',e.target.value||undefined)} rows={2} className="w-full bg-surface-2 border border-border-subtle rounded px-1.5 py-1 text-[10px] font-mono text-text-primary focus:border-accent-blue/40 focus:outline-none resize-none"/></label>
      <div className="flex gap-2 mt-2">
        <button onClick={onSave} className="flex-1 flex items-center justify-center gap-1 py-1.5 bg-accent-blue/15 border border-accent-blue/30 rounded text-accent-blue text-[10px] font-mono font-semibold hover:bg-accent-blue/25 transition-colors"><Save className="w-3 h-3"/>Save</button>
        <button onClick={onCancel} className="flex-1 py-1.5 bg-surface-2 border border-border-subtle rounded text-text-muted text-[10px] font-mono hover:text-text-primary transition-colors">Cancel</button>
      </div>
    </div>
  );
}

// ─── Detail panel ─────────────────────────────────────────────────────────────
function DetailPanel({ cellId, allCells, edits, setEdits, onNavigate }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({});
  const cell = allCells.find(c => c.grid_id === cellId);
  const merged = cell ? { ...cell, ...(edits[cellId]||{}) } : null;
  const spread = cellId ? OBJ2_SPREAD[cellId] : null;
  const tier = merged ? getRiskTier(merged.fire_risk_score) : null;
  const crown = merged ? deriveCrownFire(merged.canopy_cover_pct, merged.canopy_base_height_m, merged.canopy_bulk_density) : 'none';
  const startEdit = () => { setDraft(edits[cellId]||{}); setEditing(true); };
  const cancelEdit = () => { setDraft({}); setEditing(false); };
  const saveEdit = () => { const n={...edits}; const c=Object.fromEntries(Object.entries(draft).filter(([,v])=>v!==undefined)); if(Object.keys(c).length) n[cellId]=c; else delete n[cellId]; setEdits(n); localStorage.setItem('fireMapEdits',JSON.stringify(n)); setEditing(false); };
  const exportJSON = () => {
    if(!merged) return;
    const p={region:merged.lat>36?'california':'texas',grid_ids:[merged.grid_id],features:{temperature_2m:[merged.temperature_2m],relative_humidity_2m:[merged.relative_humidity_2m],wind_speed_10m:[merged.wind_speed_10m],wind_direction_10m:[merged.wind_direction_10m],precipitation:[merged.precipitation],soil_moisture_0_to_7cm:[merged.soil_moisture_0_to_7cm],vpd:[merged.vpd],elevation_m:[merged.elevation_m],slope_degrees:[merged.slope_degrees],aspect_degrees:[merged.aspect_degrees],canopy_cover_pct:[merged.canopy_cover_pct],canopy_base_height_m:[merged.canopy_base_height_m],canopy_bulk_density:[merged.canopy_bulk_density],fuel_model_fbfm40:[merged.fuel_model_fbfm40]},_meta:{exported_at:new Date().toISOString(),source:'fire-map-manual-override'}};
    const b=new Blob([JSON.stringify(p,null,2)],{type:'application/json'}); const u=URL.createObjectURL(b); const a=document.createElement('a'); a.href=u; a.download=`predict_${merged.grid_id.slice(0,8)}.json`; a.click(); URL.revokeObjectURL(u);
  };

  if (!cell) return <div className="flex flex-col items-center justify-center h-full gap-4 text-text-muted px-6"><Crosshair className="w-10 h-10 opacity-15"/><span className="text-[11px] text-center leading-relaxed opacity-50">Click a cell on the map<br/>to inspect details</span></div>;

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className={`px-4 py-3 border-b border-border-subtle ${tier==='CRITICAL'?'glow-critical bg-risk-critical/5':''}`}>
        <div className="flex items-start justify-between gap-2 mb-1"><div><div className="text-text-primary text-[13px] font-semibold leading-tight">{merged.name || merged.grid_id?.slice(0,12)}</div><div className="text-text-muted text-[9px] font-mono mt-0.5">{merged.grid_id}</div></div><RiskBadge tier={tier}/></div>
        <div className="text-[10px] font-mono text-text-muted">{fmt(merged.lat, 4)}°N, {fmt(Math.abs(merged.lon), 4)}°W</div>
        {edits[cellId]&&<div className="mt-1 flex items-center gap-1 text-[9px] font-mono text-purple-400"><Pencil className="w-2.5 h-2.5"/>Overrides active</div>}
      </div>

      {editing ? <EditForm cell={cell} vals={draft} onChange={(k,v)=>setDraft(d=>({...d,[k]:v}))} onSave={saveEdit} onCancel={cancelEdit}/> : <>
        <div className="px-4 py-3 border-b border-border-subtle">
          <div className="flex items-center gap-1.5 mb-2"><Activity className="w-3 h-3 text-accent-blue"/><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Ignition Risk</span></div>
          <div className="flex justify-between items-center mb-1"><span className="text-[10px] font-mono text-text-muted">P(ignition)</span><span className={`text-[14px] font-mono font-bold ${TIER_COLORS[tier].text}`}>{fmt((merged.fire_risk_score||0)*100, 1)}%</span></div>
          <div className="h-2 bg-surface-3 rounded-full overflow-hidden"><div className="h-full rounded-full transition-all duration-500" style={{width:`${(merged.fire_risk_score||0)*100}%`,background:`linear-gradient(90deg,${TIER_COLORS.LOW.fill},${TIER_COLORS[tier].fill})`,boxShadow:`0 0 8px ${TIER_COLORS[tier].glow}`}}/></div>
        </div>
        <div className="px-4 py-3 border-b border-border-subtle">
          <div className="flex items-center gap-1.5 mb-2"><Thermometer className="w-3 h-3 text-accent-orange"/><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Weather</span></div>
          <div className="divide-y divide-border-subtle/30"><Row icon={Thermometer} label="Temperature" value={merged.temperature_2m} unit="°C"/><Row icon={Droplets} label="Humidity" value={merged.relative_humidity_2m} unit="%"/><Row icon={Wind} label="Wind" value={merged.wind_speed_10m != null ? `${fmt(merged.wind_speed_10m)} km/h ${compassLabel(merged.wind_direction_10m)}` : null}/><Row icon={Activity} label="VPD" value={merged.vpd} unit="kPa"/><Row icon={Droplets} label="Soil Moisture" value={merged.soil_moisture_0_to_7cm} unit="m³/m³"/></div>
        </div>
        <div className="px-4 py-3 border-b border-border-subtle">
          <div className="flex items-center gap-1.5 mb-2"><TreePine className="w-3 h-3 text-green-500"/><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Terrain & Canopy</span></div>
          <div className="divide-y divide-border-subtle/30"><Row icon={Mountain} label="Elevation" value={merged.elevation_m} unit="m"/><Row icon={Activity} label="Slope" value={merged.slope_degrees} unit="°"/><Row icon={TreePine} label="Canopy" value={merged.canopy_cover_pct} unit="%"/><Row icon={Layers} label="Fuel" value={merged.fuel_model_fbfm40}/><Row icon={TreePine} label="Vegetation" value={merged.vegetation_type}/></div>
          {crown!=='none'&&<div className={`mt-2 px-2.5 py-2 rounded-lg border ${crown==='active'?'bg-risk-critical/8 border-risk-critical/25':'bg-risk-high/8 border-risk-high/25'}`}><span className={`text-[10px] font-mono font-semibold ${crown==='active'?'text-risk-critical':'text-risk-high'}`}>Crown Fire: {crown.toUpperCase()}</span></div>}
        </div>
        {merged.fire_detected_binary===1&&<div className="mx-4 my-2 px-3 py-2.5 bg-risk-critical/8 border border-risk-critical/25 rounded-lg glow-critical"><div className="flex items-center gap-2"><Flame className="w-3.5 h-3.5 text-risk-critical"/><span className="text-[10px] font-mono font-bold text-risk-critical">ACTIVE FIRE — {merged.active_fire_count} hotspots</span></div>{merged.mean_frp>0&&<div className="text-[9px] font-mono text-text-muted mt-1">FRP: {merged.mean_frp} MW</div>}</div>}
        {spread&&<div className="px-4 py-3 border-b border-border-subtle"><div className="flex items-center gap-1.5 mb-2"><Navigation className="w-3 h-3 text-accent-orange"/><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Spread Sim</span></div><div className="grid grid-cols-2 gap-x-3 gap-y-1">{[['Rate',`${spread.spread_rate_m_per_min} m/min`],['Area',`${spread.spread_area_km2} km²`],['Contain.',`${(spread.containment_probability*100).toFixed(0)}%`],['Wind',`${spread.wind_speed_m_s} m/s ${compassLabel(spread.wind_direction_deg)}`]].map(([k,v])=><div key={k} className="flex justify-between"><span className="text-[10px] font-mono text-text-muted">{k}</span><span className="text-[10px] font-mono text-text-primary">{v}</span></div>)}</div>{spread.notes&&<div className="mt-2 text-[9px] font-mono text-text-muted leading-relaxed">{spread.notes}</div>}</div>}
        {merged.notes&&<div className="px-4 py-3 border-b border-border-subtle"><div className="flex items-center gap-1.5 mb-1"><Info className="w-3 h-3 text-accent-blue"/><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Notes</span></div><div className="text-[10px] font-mono text-text-muted leading-relaxed">{merged.notes}</div></div>}
        <div className="px-4 py-3 space-y-2">
          <button onClick={startEdit} className="w-full flex items-center justify-between px-3 py-2.5 bg-accent-blue/8 border border-accent-blue/25 rounded-lg text-accent-blue hover:bg-accent-blue/15 transition-colors"><span className="flex items-center gap-1.5"><Pencil className="w-3 h-3"/><span className="text-[10px] font-mono font-semibold">Edit Features</span></span><ChevronRight className="w-3 h-3"/></button>
          <button onClick={exportJSON} className="w-full flex items-center justify-between px-3 py-2.5 bg-surface-2 border border-border-subtle rounded-lg text-text-secondary hover:text-text-primary hover:bg-surface-3 transition-colors"><span className="flex items-center gap-1.5"><Download className="w-3 h-3"/><span className="text-[10px] font-mono">Export JSON</span></span><ChevronRight className="w-3 h-3"/></button>
          {edits[cellId]&&<button onClick={()=>{const n={...edits};delete n[cellId];setEdits(n);localStorage.setItem('fireMapEdits',JSON.stringify(n));}} className="w-full flex items-center justify-center gap-1.5 px-3 py-2 text-text-muted hover:text-risk-critical text-[10px] font-mono transition-colors"><X className="w-3 h-3"/>Reset</button>}
          {spread&&<button onClick={()=>onNavigate?.('reports')} className="w-full flex items-center justify-between px-3 py-2.5 bg-surface-2 border border-border-subtle rounded-lg text-text-secondary hover:text-text-primary transition-colors"><span className="flex items-center gap-1.5"><FileText className="w-3 h-3"/><span className="text-[10px] font-mono">Incident Report</span></span><ChevronRight className="w-3 h-3"/></button>}
        </div>
      </>}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function FireMap({ onNavigate }) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [layers, setLayers] = useState({ colorLayer: 'risk', wind: true, crown: false, spread: false });
  const [sel, setSel] = useState(null);
  const [res, setRes] = useState('64km');
  const [mapStyle, setMapStyle] = useState('auto'); // 'auto' | 'dark' | 'light' | 'satellite'
  const [showQR, setShowQR] = useState(false);
  const [edits, setEdits] = useState(() => { try { return JSON.parse(localStorage.getItem('fireMapEdits')||'{}'); } catch { return {}; } });

  const [liveCells, setLiveCells] = useState(null);
  const [liveSpread, setLiveSpread] = useState(null); // {ignition_cell, neighbor_burn_probabilities, spread_direction_deg, ...}

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
          setLiveCells([...ca.cells.map(c => normalizeCell({ ...c, region: 'california' })), ...tx.cells.map(c => normalizeCell({ ...c, region: 'texas' }))]);
        }
      } catch { /* backend offline — fall back to mock */ }
      // Fetch live spread simulations
      try {
        const spreadMap = {};
        for (const region of ['california', 'texas']) {
          const res = await fetch(apiUrl(`/api/spread-simulations?region=${region}`));
          if (res.ok) {
            const data = await res.json();
            if (data.simulation && !data.simulation.fallback) spreadMap[region] = data.simulation;
          }
        }
        if (!cancelled && Object.keys(spreadMap).length > 0) setLiveSpread(spreadMap);
      } catch { /* backend offline */ }
      if (!cancelled) setLoading(false);
    }
    fetchLive();
    return () => { cancelled = true; };
  }, []);

  const [loading, setLoading] = useState(true);

  const allCells = useMemo(() => {
    if (liveCells) return liveCells;
    if (loading) return []; // Show empty map while loading — no mock flash
    return [...CALIFORNIA_CELLS_ENRICHED.map(c=>({...c,region:'california'})), ...TEXAS_CELLS_ENRICHED.map(c=>({...c,region:'texas'}))];
  }, [liveCells, loading]);

  const stats = useMemo(() => ({
    critical: allCells.filter(c=>getRiskTier(c.fire_risk_score)==='CRITICAL').length,
    high: allCells.filter(c=>getRiskTier(c.fire_risk_score)==='HIGH').length,
    activeFires: allCells.filter(c=>c.fire_detected_binary===1).length,
  }), [allCells]);

  // Spread overlay: live OBJ-2 burn probabilities per neighbor cell
  const { spreadOverlays, spreadArrows, ignitionCells } = useMemo(() => {
    if (!layers.spread) return { spreadOverlays: [], spreadArrows: [], ignitionCells: [] };
    const cellMap = {}; allCells.forEach(c => { cellMap[c.grid_id] = c; });
    const overlays = [];
    const arrows = [];
    const ignitions = [];

    if (liveSpread) {
      // Use live OBJ-2 simulation data
      Object.entries(liveSpread).forEach(([region, sim]) => {
        const ign = cellMap[sim.ignition_cell];
        if (ign) ignitions.push({ ...ign, spreadDir: sim.spread_direction_deg, speedKmh: sim.spread_speed_kmh, crownStatus: sim.crown_fire_status });

        const burnProbs = sim.neighbor_burn_probabilities || {};
        Object.entries(burnProbs).forEach(([cellId, prob]) => {
          const target = cellMap[cellId];
          if (!target) return;
          overlays.push({ cellId, lat: target.lat, lon: target.lon, prob, region });
          if (ign) arrows.push({ key: `${sim.ignition_cell}-${cellId}`, from: [ign.lat, ign.lon], to: [target.lat, target.lon], prob });
        });
      });
    } else {
      // Fallback to mock OBJ2_SPREAD
      Object.values(OBJ2_SPREAD).forEach(sim => {
        const s = cellMap[sim.source_cell]; if (!s) return;
        sim.affected_cells.forEach(id => { if (id === sim.source_cell) return; const t = cellMap[id]; if (t) arrows.push({ key: `${sim.source_cell}-${id}`, from: [s.lat, s.lon], to: [t.lat, t.lon], prob: 0.5 }); });
      });
    }
    return { spreadOverlays: overlays, spreadArrows: arrows, ignitionCells: ignitions };
  }, [layers.spread, allCells, liveSpread]);

  const selCell = sel ? allCells.find(c => c.grid_id === sel) : null;

  return (
    <div className="flex h-full overflow-hidden bg-surface-0">
      {/* Map area */}
      <div className="flex-1 relative min-w-0">
        <MapContainer center={DEFAULT_CENTER} zoom={DEFAULT_ZOOM} zoomControl={false}
          style={{ height: '100%', width: '100%', background: isDark ? '#0c1117' : '#f5f1eb' }}
          className="rounded-none">

          {(() => { const ts = mapStyle === 'auto' ? theme : mapStyle; return (
            <TileLayer key={ts} url={TILES[ts] || TILES.dark} attribution={TILE_ATTRS[ts] || TILE_ATTRS.dark} maxZoom={18} />
          ); })()}

          {/* Fly to selected cell */}
          {selCell && <FlyTo lat={selCell.lat} lon={selCell.lon} />}

          {/* Hex cells — real H3 boundaries when available, fallback to approximate */}
          {allCells.map(cell => {
            const m = edits[cell.grid_id] ? { ...cell, ...edits[cell.grid_id] } : cell;
            const fill = hexFill(m, layers.colorLayer);
            const isSel = cell.grid_id === sel;
            const inState = cell.in_state !== false;
            const boundary = cell.hex_boundary || hexBoundary(cell.lat, cell.lon, hexRadius(cell));
            return (
              <Polygon key={cell.grid_id} positions={boundary}
                pathOptions={{
                  fillColor: fill, fillOpacity: inState ? (isDark ? 0.55 : 0.45) : 0.15,
                  color: isSel ? '#3b82f6' : isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.2)',
                  weight: isSel ? 2.5 : 1,
                  opacity: isSel ? 1 : (inState ? 0.7 : 0.3),
                }}
                eventHandlers={{ click: () => setSel(p => p === cell.grid_id ? null : cell.grid_id) }}>
                <Tooltip sticky className={isDark ? 'leaflet-tooltip-dark' : 'leaflet-tooltip-light'}>
                  <div className="font-mono">
                    <div className="text-xs font-semibold">{cell.name || cell.grid_id?.slice(0,12)}</div>
                    <div className="text-[10px] opacity-80">
                      Risk: {fmt(m.fire_risk_score * 100, 0)}% — {getRiskTier(m.fire_risk_score)}
                      {m.fire_detected_binary===1&&<span className="text-red-400 font-bold ml-1">FIRE</span>}
                    </div>
                  </div>
                </Tooltip>
              </Polygon>
            );
          })}

          {/* Fire pulse rings */}
          {allCells.filter(c=>c.fire_detected_binary===1).map(c=>(
            <CircleMarker key={`fp-${c.grid_id}`} center={[c.lat,c.lon]} radius={18}
              pathOptions={{color:'#ff4400',weight:2,fillColor:'#ff4400',fillOpacity:0.1,dashArray:'4,4'}}
              className="fire-pulse-ring" interactive={false} />
          ))}

          {/* Live spread: burn probability overlays on neighbor cells */}
          {spreadOverlays.map(o => (
            <CircleMarker key={`burn-${o.cellId}`} center={[o.lat, o.lon]}
              radius={Math.max(8, o.prob * 30)}
              pathOptions={{
                color: o.prob >= 0.5 ? '#ef4444' : o.prob >= 0.2 ? '#f59e0b' : '#10b981',
                weight: 1.5,
                fillColor: o.prob >= 0.5 ? '#ef4444' : o.prob >= 0.2 ? '#f59e0b' : '#10b981',
                fillOpacity: 0.15 + o.prob * 0.35,
              }}
              className="spread-burn-overlay" interactive={false} />
          ))}

          {/* Ignition cell pulse animation */}
          {ignitionCells.map(ign => (
            <CircleMarker key={`ign-${ign.grid_id}`} center={[ign.lat, ign.lon]}
              radius={22}
              pathOptions={{
                color: '#ef4444', weight: 3, fillColor: '#ef4444', fillOpacity: 0.25,
                dashArray: '6,3',
              }}
              className="ignition-pulse" interactive={false} />
          ))}

          {/* Spread direction arrows (ignition → neighbor) */}
          {spreadArrows.map(a=>(
            <Polyline key={a.key} positions={[a.from,a.to]}
              pathOptions={{
                color: a.prob >= 0.5 ? '#ef4444' : '#fb923c',
                weight: 1.5 + a.prob * 2,
                dashArray: '6,4',
                opacity: 0.4 + a.prob * 0.5,
              }} interactive={false} />
          ))}

          {/* Wind arrows */}
          {layers.wind && allCells.filter(c=>c.wind_direction_10m!=null).map(c=>(
            <Marker key={`w-${c.grid_id}`} position={[c.lat,c.lon]}
              icon={makeWindIcon(edits[c.grid_id]?.wind_direction_10m ?? c.wind_direction_10m, edits[c.grid_id]?.wind_speed_10m ?? c.wind_speed_10m, isDark)}
              interactive={false} />
          ))}

          {/* Crown badges */}
          {layers.crown && allCells.map(c=>{
            const m = edits[c.grid_id] ? {...c,...edits[c.grid_id]} : c;
            const cr = deriveCrownFire(m.canopy_cover_pct,m.canopy_base_height_m,m.canopy_bulk_density);
            if(cr==='none') return null;
            return <Marker key={`cr-${c.grid_id}`} position={[c.lat,c.lon]} icon={makeCrownIcon(cr)} interactive={false}/>;
          })}
        </MapContainer>

        {/* Floating controls */}
        <LayerBar layers={layers} setLayers={setLayers} stats={stats} resolution={res} setResolution={setRes} mapStyle={mapStyle} setMapStyle={setMapStyle} onQR={() => setShowQR(true)} />
        <Legend layers={layers} />
        {showQR && <QRModal onClose={() => setShowQR(false)} />}
      </div>

      {/* Mobile backdrop — tap outside to close panel */}
      {sel && (
        <div
          className="md:hidden fixed inset-0 bg-black/40 z-[999]"
          onClick={() => setSel(null)}
        />
      )}

      {/* Detail panel — fixed overlay on mobile, static sidebar on desktop */}
      <div className={`
        fixed inset-y-0 right-0 md:static md:inset-auto
        w-[280px] flex-shrink-0 border-l border-border-subtle bg-surface-1 flex flex-col
        z-[1000] transition-transform duration-300
        ${sel ? 'translate-x-0' : 'translate-x-full md:translate-x-0'}
      `}>
        <div className="px-4 py-2.5 border-b border-border-subtle flex items-center gap-2 flex-shrink-0">
          <MapPin className="w-3.5 h-3.5 text-text-muted"/>
          <span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Cell Detail</span>
          {sel && (
            <button onClick={() => setSel(null)} className="ml-auto flex items-center gap-1 text-[9px] font-mono text-text-muted hover:text-text-primary transition-colors">
              <X className="w-3 h-3" />Close
            </button>
          )}
        </div>
        <div className="flex-1 overflow-hidden"><DetailPanel cellId={sel} allCells={allCells} edits={edits} setEdits={setEdits} onNavigate={onNavigate}/></div>
      </div>
    </div>
  );
}
