import { useState, useMemo, useEffect } from 'react';
import {
  Flame, Wind, Thermometer, Droplets, Activity, TreePine,
  FileText, ChevronRight, Info, Crosshair, MapPin, Mountain,
  Pencil, Save, Download, X, EyeOff, Navigation, Layers, BarChart3,
} from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Polygon, Polyline, CircleMarker, Tooltip, Marker, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import { useTheme } from '../ui/ThemeProvider';
import { CALIFORNIA_CELLS_ENRICHED, TEXAS_CELLS_ENRICHED, getRiskTier } from '../../data/mockGridData';
import { OBJ2_SPREAD } from '../../data/mockMapData';
import {
  hexBoundary, compassLabel,
  riskColor, moistureColor, intensityColor, windSpeedOpacity,
  deriveCrownFire, TIER_COLORS,
  CA_OUTLINE, TX_OUTLINE,
  generateFireZoomCells,
} from './mapHelpers';

// ─── Config ───────────────────────────────────────────────────────────────────
const TILES = {
  dark:  'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  light: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
};
const TILE_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>';
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
  return <div className="flex items-center justify-between py-[3px]"><span className="flex items-center gap-1.5 text-text-muted"><I className="w-3 h-3"/><span className="text-[10px] font-mono">{label}</span></span><span className="text-[10px] font-mono text-text-primary">{value}{unit?` ${unit}`:''}</span></div>;
}

const US_STATES_URL = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json';
const MONITORING_STATES = new Set(['California', 'Texas']);

// ─── Create a lower pane for state boundaries (so hex cells stay on top) ─────
function StatesPane() {
  const map = useMap();
  useEffect(() => {
    if (!map.getPane('statesPane')) {
      map.createPane('statesPane');
      map.getPane('statesPane').style.zIndex = 350;
    }
  }, [map]);
  return null;
}

// ─── Fly to selected cell ─────────────────────────────────────────────────────
function FlyTo({ lat, lon }) {
  const map = useMap();
  useEffect(() => {
    if (lat != null && lon != null) map.flyTo([lat, lon], Math.max(map.getZoom(), 7), { duration: 0.6 });
  }, [lat, lon, map]);
  return null;
}

// ─── Fly to GeoJSON feature bounds ────────────────────────────────────────────
function FlyToBounds({ geojson, name }) {
  const map = useMap();
  useEffect(() => {
    if (!geojson || !name) return;
    const feature = geojson.features.find(f => f.properties.name === name);
    if (!feature) return;
    const layer = L.geoJSON(feature);
    map.flyToBounds(layer.getBounds().pad(0.1), { duration: 0.8 });
  }, [geojson, name, map]);
  return null;
}

// ─── Wind arrow icon factory ──────────────────────────────────────────────────
function makeWindIcon(dir, speed, dark = true) {
  const rot = (dir + 180) % 360;
  const op = windSpeedOpacity(speed);
  const color = dark ? '#e0f2fe' : '#1e3a5f';
  return L.divIcon({
    html: `<div style="transform:rotate(${rot}deg);opacity:${op};width:22px;height:22px">
      <svg viewBox="0 0 22 22" width="22" height="22">
        <line x1="11" y1="18" x2="11" y2="5" stroke="${color}" stroke-width="2.2" stroke-linecap="round"/>
        <polygon points="11,2 7.5,8 14.5,8" fill="${color}"/>
      </svg></div>`,
    className: '',
    iconSize: [22, 22],
    iconAnchor: [11, 11],
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

// ─── Layer bar ────────────────────────────────────────────────────────────────
function LayerBar({ layers, setLayers, stats, resolution, setResolution }) {
  const setColor = k => setLayers(l => ({ ...l, colorLayer: k }));
  const toggle = k => setLayers(l => ({ ...l, [k]: !l[k] }));
  return (
    <div className="absolute top-3 left-3 right-[290px] z-[1000] flex items-center justify-between px-3 py-2 bg-surface-1/90 backdrop-blur-md border border-border-subtle rounded-xl shadow-xl">
      <div className="flex items-center gap-3">
        <div className="flex bg-surface-2 border border-border-subtle rounded-lg p-0.5 gap-px">
          {[{k:'risk',I:Activity,l:'Risk'},{k:'moisture',I:Droplets,l:'Moisture'},{k:'intensity',I:BarChart3,l:'Intensity'}].map(({k,I,l})=>(
            <button key={k} onClick={()=>setColor(k)} className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-mono transition-colors ${layers.colorLayer===k?'bg-surface-0 text-text-primary font-semibold shadow-sm':'text-text-muted hover:text-text-secondary'}`}><I className="w-3 h-3"/>{l}</button>
          ))}
        </div>
        <div className="flex items-center gap-1 pl-3 border-l border-border-subtle">
          {[{k:'wind',I:Wind,l:'Wind'},{k:'crown',I:TreePine,l:'Crown'},{k:'spread',I:Navigation,l:'Spread'}].map(({k,I,l})=>(
            <button key={k} onClick={()=>toggle(k)} className={`flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-mono border transition-colors ${layers[k]?'bg-accent-blue/10 text-accent-blue border-accent-blue/30':'text-text-muted border-transparent hover:text-text-secondary'}`}>
              {layers[k]?<I className="w-3 h-3"/>:<EyeOff className="w-3 h-3 opacity-40"/>}{l}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5 pl-3 border-l border-border-subtle">
          {stats.critical>0&&<span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/10 text-risk-critical border-risk-critical/30 animate-pulse">{stats.critical} CRIT</span>}
          {stats.high>0&&<span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-high/10 text-risk-high border-risk-high/30">{stats.high} HIGH</span>}
          {stats.activeFires>0&&<span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/10 text-risk-critical border-risk-critical/30"><Flame className="w-2.5 h-2.5 inline mr-0.5"/>{stats.activeFires}</span>}
        </div>
      </div>
      <div className="flex bg-surface-2 border border-border-subtle rounded-lg p-0.5 gap-px">
        {['64km','22km'].map(r=>(
          <button key={r} onClick={()=>setResolution(r)} className={`px-2 py-0.5 rounded-md text-[9px] font-mono transition-colors ${resolution===r?'bg-surface-0 text-text-primary font-semibold':'text-text-muted hover:text-text-secondary'}`}>{r}</button>
        ))}
      </div>
    </div>
  );
}

// ─── Legend ────────────────────────────────────────────────────────────────────
function Legend({ layers }) {
  return (
    <div className="absolute bottom-6 left-3 z-[1000] flex items-center gap-4 px-3 py-1.5 bg-surface-1/90 backdrop-blur-md border border-border-subtle rounded-xl shadow-xl">
      {layers.colorLayer==='risk'&&<><div className="flex items-center gap-1"><div className="w-24 h-2 rounded" style={{background:'linear-gradient(90deg,#22c55e,#4ade80,#a3e635,#eab308,#f59e0b,#f97316,#ef4444,#dc2626,#991b1b)'}}/></div><div className="flex items-center gap-2 ml-1">{[['LOW','#22c55e'],['MED','#eab308'],['HIGH','#f97316'],['CRIT','#ef4444']].map(([l,c])=><div key={l} className="flex items-center gap-1"><div className="w-2 h-2 rounded-sm" style={{background:c}}/><span className="text-[9px] font-mono text-text-muted">{l}</span></div>)}</div></>}
      {layers.colorLayer==='moisture'&&<div className="flex items-center gap-1"><div className="w-24 h-2 rounded" style={{background:'linear-gradient(90deg,#78350f,#92400e,#b45309,#d97706,#a3e635,#22c55e,#059669,#047857)'}}/><span className="text-[9px] font-mono text-text-muted">Dry→Wet</span></div>}
      {layers.colorLayer==='intensity'&&<div className="flex items-center gap-1"><div className="w-24 h-2 rounded" style={{background:'linear-gradient(90deg,#422006,#fef3c7,#fbbf24,#f97316,#ef4444,#b91c1c,#7c3aed)'}}/><span className="text-[9px] font-mono text-text-muted">FRP (MW)</span></div>}
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
        <div className="flex items-start justify-between gap-2 mb-1"><div><div className="text-text-primary text-[13px] font-semibold leading-tight">{merged.name}</div><div className="text-text-muted text-[9px] font-mono mt-0.5">{merged.grid_id.slice(0,12)}...</div></div><RiskBadge tier={tier}/></div>
        <div className="text-[10px] font-mono text-text-muted">{merged.lat.toFixed(2)}°N, {Math.abs(merged.lon).toFixed(2)}°W</div>
        {edits[cellId]&&<div className="mt-1 flex items-center gap-1 text-[9px] font-mono text-purple-400"><Pencil className="w-2.5 h-2.5"/>Overrides active</div>}
      </div>

      {editing ? <EditForm cell={cell} vals={draft} onChange={(k,v)=>setDraft(d=>({...d,[k]:v}))} onSave={saveEdit} onCancel={cancelEdit}/> : <>
        <div className="px-4 py-3 border-b border-border-subtle">
          <div className="flex items-center gap-1.5 mb-2"><Activity className="w-3 h-3 text-accent-blue"/><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Ignition Risk</span></div>
          <div className="flex justify-between items-center mb-1"><span className="text-[10px] font-mono text-text-muted">P(ignition)</span><span className={`text-[14px] font-mono font-bold ${TIER_COLORS[tier].text}`}>{(merged.fire_risk_score*100).toFixed(1)}%</span></div>
          <div className="h-2 bg-surface-3 rounded-full overflow-hidden"><div className="h-full rounded-full transition-all duration-500" style={{width:`${merged.fire_risk_score*100}%`,background:'linear-gradient(90deg,#22c55e,#a3e635,#eab308,#f59e0b,#f97316,#ef4444,#dc2626)',boxShadow:`0 0 8px ${TIER_COLORS[tier].glow}`}}/></div>
        </div>
        <div className="px-4 py-3 border-b border-border-subtle">
          <div className="flex items-center gap-1.5 mb-2"><Thermometer className="w-3 h-3 text-accent-orange"/><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Weather</span></div>
          <div className="divide-y divide-border-subtle/30"><Row icon={Thermometer} label="Temperature" value={merged.temperature_2m} unit="°C"/><Row icon={Droplets} label="Humidity" value={merged.relative_humidity_2m} unit="%"/><Row icon={Wind} label="Wind" value={`${merged.wind_speed_10m} m/s ${compassLabel(merged.wind_direction_10m)}`}/><Row icon={Activity} label="VPD" value={merged.vpd} unit="kPa"/><Row icon={Droplets} label="Soil Moisture" value={merged.soil_moisture_0_to_7cm} unit="m³/m³"/></div>
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
  const [activeRegion, setActiveRegion] = useState(null);
  const [statesGeo, setStatesGeo] = useState(null);
  const [res, setRes] = useState('64km');

  // Fetch US states GeoJSON once
  useEffect(() => {
    fetch(US_STATES_URL).then(r => r.json()).then(setStatesGeo).catch(() => {});
  }, []);
  const [edits, setEdits] = useState(() => { try { return JSON.parse(localStorage.getItem('fireMapEdits')||'{}'); } catch { return {}; } });

  const allCells = useMemo(() => {
    const ca = res === '22km' ? generateFireZoomCells(CALIFORNIA_CELLS_ENRICHED, OBJ2_SPREAD) : CALIFORNIA_CELLS_ENRICHED;
    const tx = res === '22km' ? generateFireZoomCells(TEXAS_CELLS_ENRICHED, OBJ2_SPREAD) : TEXAS_CELLS_ENRICHED;
    return [...ca.map(c=>({...c,region:'california'})), ...tx.map(c=>({...c,region:'texas'}))];
  }, [res]);

  const stats = useMemo(() => ({
    critical: allCells.filter(c=>getRiskTier(c.fire_risk_score)==='CRITICAL').length,
    high: allCells.filter(c=>getRiskTier(c.fire_risk_score)==='HIGH').length,
    activeFires: allCells.filter(c=>c.fire_detected_binary===1).length,
  }), [allCells]);

  // Spread arrows (source → target polylines)
  const spreadArrows = useMemo(() => {
    if (!layers.spread) return [];
    const map = {}; allCells.forEach(c => { map[c.grid_id] = c; });
    const arr = [];
    Object.values(OBJ2_SPREAD).forEach(sim => {
      const s = map[sim.source_cell]; if (!s) return;
      sim.affected_cells.forEach(id => { if (id === sim.source_cell) return; const t = map[id]; if (t) arr.push({ key:`${sim.source_cell}-${id}`, from:[s.lat,s.lon], to:[t.lat,t.lon] }); });
    });
    return arr;
  }, [layers.spread, allCells]);

  const selCell = sel ? allCells.find(c => c.grid_id === sel) : null;

  return (
    <div className="flex h-full overflow-hidden bg-surface-0">
      {/* Map area */}
      <div className="flex-1 relative min-w-0">
        <MapContainer center={DEFAULT_CENTER} zoom={DEFAULT_ZOOM} zoomControl={false}
          minZoom={3}
          style={{ height: '100%', width: '100%', background: isDark ? '#0c1117' : '#f5f1eb' }}
          className="rounded-none">

          <TileLayer key={theme} url={TILES[theme] || TILES.dark} attribution={TILE_ATTR} maxZoom={18} />
          <StatesPane />

          {/* Mask everything outside CA & TX */}
          <Polygon positions={[
            [[-90, -180], [-90, 180], [90, 180], [90, -180]],
            CA_OUTLINE.map(([lat, lon]) => [lat, lon]),
            TX_OUTLINE.map(([lat, lon]) => [lat, lon]),
          ]} pathOptions={{
            color: 'transparent', weight: 0,
            fillColor: isDark ? '#0c1117' : '#e8e4dd',
            fillOpacity: isDark ? 0.55 : 0.5,
          }} interactive={false} />

          {/* All US states — clickable, selected one lights up */}
          {statesGeo && (
            <GeoJSON
              key={`states-${activeRegion || 'none'}-${theme}`}
              data={statesGeo}
              style={(feature) => {
                const name = feature.properties.name;
                const isActive = activeRegion === name;
                const isMonitored = MONITORING_STATES.has(name);
                return {
                  pane: 'statesPane',
                  color: isActive ? (isDark ? '#60a5fa' : '#2563eb') : isMonitored ? (isDark ? 'rgba(96,165,250,0.3)' : 'rgba(37,99,235,0.2)') : (isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'),
                  weight: isActive ? 2.5 : isMonitored ? 1.2 : 0.5,
                  fillColor: isActive ? (isDark ? '#60a5fa' : '#3b82f6') : 'transparent',
                  fillOpacity: isActive ? (isDark ? 0.15 : 0.1) : 0,
                };
              }}
              onEachFeature={(feature, layer) => {
                const name = feature.properties.name;
                layer.on('click', () => setActiveRegion(prev => prev === name ? null : name));
                layer.bindTooltip(name, {
                  sticky: false,
                  direction: 'center',
                  className: isDark ? 'leaflet-tooltip-dark' : 'leaflet-tooltip-light',
                });
              }}
            />
          )}

          {/* Fly to region or cell */}
          <FlyToBounds geojson={statesGeo} name={activeRegion} />
          {selCell && <FlyTo lat={selCell.lat} lon={selCell.lon} />}

          {/* Hex cells */}
          {allCells.map(cell => {
            const m = edits[cell.grid_id] ? { ...cell, ...edits[cell.grid_id] } : cell;
            const fill = hexFill(m, layers.colorLayer);
            const isSel = cell.grid_id === sel;
            const boundary = hexBoundary(cell.lat, cell.lon, hexRadius(cell));
            return (
              <Polygon key={cell.grid_id} positions={boundary}
                pathOptions={{
                  fillColor: fill, fillOpacity: isDark ? 0.7 : 0.6,
                  color: isSel ? '#3b82f6' : isDark ? 'rgba(0,0,0,0.6)' : 'rgba(0,0,0,0.25)',
                  weight: isSel ? 3 : 0.8,
                  opacity: isSel ? 1 : 0.6,
                }}
                eventHandlers={{ click: () => setSel(p => p === cell.grid_id ? null : cell.grid_id) }}>
                <Tooltip sticky className={isDark ? 'leaflet-tooltip-dark' : 'leaflet-tooltip-light'}>
                  <div className="font-mono">
                    <div className="text-xs font-semibold">{cell.name}</div>
                    <div className="text-[10px] opacity-80">
                      Risk: {(m.fire_risk_score*100).toFixed(0)}% — {getRiskTier(m.fire_risk_score)}
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
              className="fire-pulse-ring" />
          ))}

          {/* Spread perimeters */}
          {layers.spread && Object.values(OBJ2_SPREAD).map(sim => (
            <Polygon key={`sp-${sim.source_cell}`}
              positions={sim.perimeter_coords.map(({lat,lon})=>[lat,lon])}
              pathOptions={{color:'#fb923c',weight:1.8,dashArray:'8,5',fillColor:'#fb923c',fillOpacity:0.06}} />
          ))}

          {/* Spread direction arrows */}
          {spreadArrows.map(a=>(
            <Polyline key={a.key} positions={[a.from,a.to]}
              pathOptions={{color:'#fb923c',weight:2.5,dashArray:'6,4',opacity:0.7}} />
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
        <LayerBar layers={layers} setLayers={setLayers} stats={stats} resolution={res} setResolution={setRes} />
        <Legend layers={layers} />
      </div>

      {/* Detail panel */}
      <div className="w-[280px] flex-shrink-0 border-l border-border-subtle bg-surface-1 flex flex-col z-[1000]">
        <div className="px-4 py-2.5 border-b border-border-subtle flex items-center gap-2 flex-shrink-0">
          <MapPin className="w-3.5 h-3.5 text-text-muted"/><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Cell Detail</span>
          {sel&&<button onClick={()=>setSel(null)} className="ml-auto text-[9px] font-mono text-text-muted hover:text-text-primary">clear</button>}
        </div>
        <div className="flex-1 overflow-hidden"><DetailPanel cellId={sel} allCells={allCells} edits={edits} setEdits={setEdits} onNavigate={onNavigate}/></div>
      </div>
    </div>
  );
}
