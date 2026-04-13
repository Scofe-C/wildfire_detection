import { useState, useMemo, useCallback, useRef, useEffect, memo } from 'react';
import {
  Flame, Wind, Thermometer, Droplets, Activity, TreePine,
  FileText, ChevronRight, Info, Crosshair, MapPin, Mountain,
  ZoomIn, ZoomOut, Maximize2, Pencil, Save, Download, X,
  EyeOff, Navigation, Layers, BarChart3,
} from 'lucide-react';
import { CALIFORNIA_CELLS_ENRICHED, TEXAS_CELLS_ENRICHED, getRiskTier } from '../../data/mockGridData';
import { OBJ2_SPREAD } from '../../data/mockMapData';
import {
  project, hexPoints, hexPathD, windArrowD, spreadArrowD, compassLabel,
  riskColor, moistureColor, intensityColor, windSpeedOpacity,
  deriveCrownFire, CROWN_CFG, TIER_COLORS,
  CA_BOUNDS, TX_BOUNDS, CA_OUTLINE, TX_OUTLINE, CA_CITIES, TX_CITIES,
  generateTerrainGrid, generateFireZoomCells,
} from './mapHelpers';

// ─── Layout ───────────────────────────────────────────────────────────────────
const MAP_W = 460;
const MAP_H = 460;
const PAD = 20;
const HEX_R = 11;
const DRAG_THRESHOLD = 4;

// ─── Hex fill by layer ────────────────────────────────────────────────────────
function hexFill(cell, layer) {
  if (layer === 'moisture')  return moistureColor(cell.relative_humidity_2m);
  if (layer === 'intensity') return intensityColor(cell.mean_frp || 0);
  return riskColor(cell.fire_risk_score);
}
function clampVB(vb) {
  const w = Math.max(80, Math.min(MAP_W * 2.5, vb.w));
  const h = Math.max(80, Math.min(MAP_H * 2.5, vb.h));
  return { x: vb.x, y: vb.y, w, h };
}

// ─── Tiny components ──────────────────────────────────────────────────────────
function RiskBadge({ tier }) {
  const c = { CRITICAL:'bg-risk-critical/20 text-risk-critical border-risk-critical/40', HIGH:'bg-risk-high/20 text-risk-high border-risk-high/40', MEDIUM:'bg-risk-medium/20 text-risk-medium border-risk-medium/40', LOW:'bg-risk-low/20 text-risk-low border-risk-low/40' }[tier];
  return <span className={`text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded border leading-none ${c}`}>{tier}</span>;
}
function Row({ icon: I, label, value, unit }) {
  return <div className="flex items-center justify-between py-[3px]"><span className="flex items-center gap-1.5 text-text-muted"><I className="w-3 h-3" /><span className="text-[10px] font-mono">{label}</span></span><span className="text-[10px] font-mono text-text-primary">{value}{unit ? ` ${unit}` : ''}</span></div>;
}

// ─── SVG filters ──────────────────────────────────────────────────────────────
function Defs() {
  return (
    <defs>
      {/* Soft drop shadow for hex cells */}
      <filter id="hex-lift" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="0.6" stdDeviation="1.2" floodColor="#000" floodOpacity="0.45" />
      </filter>
      {/* Warm fire glow */}
      <filter id="fire-glow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="b" />
        <feFlood floodColor="#ff4400" floodOpacity="0.3" />
        <feComposite in2="b" operator="in" result="g" />
        <feMerge><feMergeNode in="g" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
      {/* Selection ring glow */}
      <filter id="sel-glow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="b" />
        <feFlood floodColor="#60a5fa" floodOpacity="0.5" />
        <feComposite in2="b" operator="in" result="g" />
        <feMerge><feMergeNode in="g" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
    </defs>
  );
}

// ─── Layer control bar ────────────────────────────────────────────────────────
function LayerBar({ layers, setLayers, stats, resolution, setResolution }) {
  const setColor = k => setLayers(l => ({ ...l, colorLayer: k }));
  const toggle = k => setLayers(l => ({ ...l, [k]: !l[k] }));
  const colorItems = [
    { k: 'risk',      icon: Activity,  label: 'Risk' },
    { k: 'moisture',  icon: Droplets,  label: 'Moisture' },
    { k: 'intensity', icon: BarChart3, label: 'Intensity' },
  ];
  const overlays = [
    { k: 'wind',    icon: Wind,       label: 'Wind' },
    { k: 'crown',   icon: TreePine,   label: 'Crown' },
    { k: 'spread',  icon: Navigation, label: 'Spread' },
    { k: 'terrain', icon: Mountain,   label: 'Terrain' },
  ];

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-surface-1 border-b border-border-subtle flex-shrink-0">
      <div className="flex items-center gap-3">
        {/* Color layers */}
        <div className="flex bg-surface-2 border border-border-subtle rounded-md p-0.5 gap-px">
          {colorItems.map(({ k, icon: Ic, label }) => (
            <button key={k} onClick={() => setColor(k)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-mono transition-colors
                ${layers.colorLayer === k ? 'bg-surface-0 text-text-primary font-semibold shadow-sm' : 'text-text-muted hover:text-text-secondary'}`}>
              <Ic className="w-3 h-3" />{label}
            </button>
          ))}
        </div>
        {/* Overlay toggles */}
        <div className="flex items-center gap-1 pl-3 border-l border-border-subtle">
          {overlays.map(({ k, icon: Ic, label }) => (
            <button key={k} onClick={() => toggle(k)}
              className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono border transition-colors
                ${layers[k] ? 'bg-accent-blue/10 text-accent-blue border-accent-blue/30' : 'text-text-muted border-transparent hover:text-text-secondary'}`}>
              {layers[k] ? <Ic className="w-3 h-3" /> : <EyeOff className="w-3 h-3 opacity-40" />}{label}
            </button>
          ))}
        </div>
        {/* Stats */}
        <div className="flex items-center gap-1.5 pl-3 border-l border-border-subtle">
          {stats.critical > 0 && <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/10 text-risk-critical border-risk-critical/30 animate-pulse">{stats.critical} CRIT</span>}
          {stats.high > 0 && <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-high/10 text-risk-high border-risk-high/30">{stats.high} HIGH</span>}
          {stats.activeFires > 0 && <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/10 text-risk-critical border-risk-critical/30"><Flame className="w-2.5 h-2.5 inline mr-0.5" />{stats.activeFires}</span>}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex bg-surface-2 border border-border-subtle rounded-md p-0.5 gap-px">
          {['64km', '22km'].map(r => (
            <button key={r} onClick={() => setResolution(r)}
              className={`px-2 py-0.5 rounded text-[9px] font-mono transition-colors ${resolution === r ? 'bg-surface-0 text-text-primary font-semibold' : 'text-text-muted hover:text-text-secondary'}`}>
              {r}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Zoom controls (floating) ─────────────────────────────────────────────────
function ZoomBtns({ onIn, onOut, onFit, pct }) {
  const btn = "p-2 text-text-muted hover:text-text-primary hover:bg-white/5 transition-colors";
  return (
    <div className="absolute bottom-4 right-4 flex flex-col gap-1.5 z-10">
      <div className="flex flex-col bg-surface-1/90 backdrop-blur border border-border-subtle rounded-xl shadow-xl overflow-hidden">
        <button onClick={onIn} className={btn} title="Zoom in"><ZoomIn className="w-4 h-4" /></button>
        <div className="text-[8px] font-mono text-text-muted text-center py-0.5 border-y border-border-subtle/40">{pct}%</div>
        <button onClick={onOut} className={btn} title="Zoom out"><ZoomOut className="w-4 h-4" /></button>
      </div>
      <button onClick={onFit} className={`${btn} bg-surface-1/90 backdrop-blur border border-border-subtle rounded-xl shadow-xl`} title="Reset view">
        <Maximize2 className="w-4 h-4" />
      </button>
    </div>
  );
}

// ─── Region Map (the core SVG map) ────────────────────────────────────────────
const RegionMap = memo(function RegionMap({ cells, bounds, label, layers, selectedId, spreadData, onSelect, edits }) {
  const iW = MAP_W - PAD * 2, iH = MAP_H - PAD * 2;
  const init = { x: 0, y: 0, w: MAP_W, h: MAP_H };
  const [vb, setVB] = useState(init);
  const svgRef = useRef(null);
  const pan = useRef({});
  const [hov, setHov] = useState(null);

  const outline = label === 'California' ? CA_OUTLINE : TX_OUTLINE;
  const cities  = label === 'California' ? CA_CITIES  : TX_CITIES;
  const zoom = vb.w / MAP_W;
  const hexR = Math.max(5, Math.min(20, HEX_R / Math.sqrt(zoom)));

  // Pre-compute projections
  const proj = useMemo(() => cells.map(c => {
    const { x, y } = project(c.lat, c.lon, bounds, iW, iH);
    return { ...c, cx: x + PAD, cy: y + PAD };
  }), [cells, bounds, iW, iH]);

  const outPts = useMemo(() => outline.map(([lat, lon]) => {
    const { x, y } = project(lat, lon, bounds, iW, iH);
    return `${x + PAD},${y + PAD}`;
  }).join(' '), [outline, bounds, iW, iH]);

  const cityPts = useMemo(() => cities.map(({ name, lat, lon }) => {
    const { x, y } = project(lat, lon, bounds, iW, iH);
    return { name, cx: x + PAD, cy: y + PAD };
  }), [cities, bounds, iW, iH]);

  // Terrain (only computed when layer is on)
  const terrain = useMemo(() => layers.terrain ? generateTerrainGrid(bounds, MAP_W, MAP_H, PAD, 32) : [], [bounds, layers.terrain]);

  // Spread
  const srcSet = useMemo(() => new Set(Object.keys(spreadData)), [spreadData]);
  const affSet = useMemo(() => { const s = new Set(); Object.values(spreadData).forEach(sim => sim.affected_cells.forEach(id => s.add(id))); return s; }, [spreadData]);
  const peris = useMemo(() => !layers.spread ? [] : Object.values(spreadData).filter(s => cells.some(c => c.grid_id === s.source_cell)).map(s => ({
    id: s.source_cell,
    pts: s.perimeter_coords.map(({ lat, lon }) => { const { x, y } = project(lat, lon, bounds, iW, iH); return `${x + PAD},${y + PAD}`; }).join(' '),
  })), [layers.spread, spreadData, cells, bounds, iW, iH]);

  const spArrows = useMemo(() => {
    if (!layers.spread) return [];
    const map = {}; proj.forEach(c => { map[c.grid_id] = c; });
    const arr = [];
    Object.values(spreadData).forEach(sim => {
      const s = map[sim.source_cell]; if (!s) return;
      sim.affected_cells.forEach(id => { if (id === sim.source_cell) return; const t = map[id]; if (!t) return; const a = spreadArrowD(s.cx, s.cy, t.cx, t.cy); if (a) arr.push({ k: `${sim.source_cell}-${id}`, ...a }); });
    });
    return arr;
  }, [layers.spread, spreadData, proj]);

  // Batched hex paths by fill color
  const hexBatches = useMemo(() => {
    const g = {};
    proj.forEach(c => {
      const m = edits[c.grid_id] ? { ...c, ...edits[c.grid_id] } : c;
      const f = hexFill(m, layers.colorLayer);
      if (!g[f]) g[f] = [];
      g[f].push(hexPathD(c.cx, c.cy, hexR));
    });
    return g;
  }, [proj, layers.colorLayer, hexR, edits]);

  // Find nearest cell to client coords
  const nearest = useCallback((cx, cy) => {
    const svg = svgRef.current; if (!svg) return null;
    const r = svg.getBoundingClientRect();
    const sx = vb.x + ((cx - r.left) / r.width) * vb.w;
    const sy = vb.y + ((cy - r.top) / r.height) * vb.h;
    let best = null, bestD = hexR * 3.2;
    proj.forEach(c => { const d = Math.hypot(c.cx - sx, c.cy - sy); if (d < bestD) { bestD = d; best = c; } });
    return best;
  }, [vb, proj, hexR]);

  // ── Pointer handlers (Google Maps style: click-hold-drag to pan) ──
  const onDown = useCallback(e => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pan.current = { sx: e.clientX, sy: e.clientY, lx: e.clientX, ly: e.clientY, dragged: false };
  }, []);

  const onMove = useCallback(e => {
    const p = pan.current;
    // Hover (no pointer down)
    if (p.sx === undefined) { const c = nearest(e.clientX, e.clientY); setHov(c ? c.grid_id : null); return; }
    // Check threshold
    if (!p.dragged && Math.hypot(e.clientX - p.sx, e.clientY - p.sy) < DRAG_THRESHOLD) return;
    p.dragged = true;
    const svg = svgRef.current; if (!svg) return;
    const r = svg.getBoundingClientRect();
    setVB(v => {
      const dx = (e.clientX - p.lx) * (v.w / r.width);
      const dy = (e.clientY - p.ly) * (v.h / r.height);
      p.lx = e.clientX; p.ly = e.clientY;
      return { ...v, x: v.x - dx, y: v.y - dy };
    });
  }, [nearest]);

  const onUp = useCallback(e => {
    const wasDrag = pan.current.dragged;
    pan.current = {};
    if (!wasDrag) { const c = nearest(e.clientX, e.clientY); onSelect(c ? c.grid_id : null); }
  }, [nearest, onSelect]);

  // Wheel zoom
  useEffect(() => {
    const svg = svgRef.current; if (!svg) return;
    const handler = e => {
      e.preventDefault();
      const f = e.deltaY > 0 ? 1.12 : 1 / 1.12;
      const r = svg.getBoundingClientRect();
      setVB(v => {
        const mx = v.x + ((e.clientX - r.left) / r.width) * v.w;
        const my = v.y + ((e.clientY - r.top) / r.height) * v.h;
        const nw = v.w * f, nh = v.h * f;
        return clampVB({ x: mx - (mx - v.x) * (nw / v.w), y: my - (my - v.y) * (nh / v.h), w: nw, h: nh });
      });
    };
    svg.addEventListener('wheel', handler, { passive: false });
    return () => svg.removeEventListener('wheel', handler);
  }, []);

  const zIn  = () => setVB(v => clampVB({ x: v.x + v.w * .1, y: v.y + v.h * .1, w: v.w * .8, h: v.h * .8 }));
  const zOut = () => setVB(v => clampVB({ x: v.x - v.w * .125, y: v.y - v.h * .125, w: v.w * 1.25, h: v.h * 1.25 }));
  const fit  = () => setVB(init);
  const showLabels = zoom < 0.5;
  const hovCell = hov ? proj.find(c => c.grid_id === hov) : null;

  return (
    <div className="relative flex-1 flex flex-col items-center min-w-0">
      <div className="text-[10px] font-mono text-text-muted uppercase tracking-[0.25em] mb-1 mt-1 font-semibold">{label}</div>
      <div className="relative flex-1 w-full" style={{ maxHeight: MAP_H + 4 }}>
        <svg ref={svgRef}
          viewBox={`${vb.x.toFixed(1)} ${vb.y.toFixed(1)} ${vb.w.toFixed(1)} ${vb.h.toFixed(1)}`}
          className="w-full h-full rounded-xl border border-border-subtle/60 shadow-lg"
          style={{ cursor: pan.current.dragged ? 'grabbing' : hov ? 'pointer' : 'grab', background: '#0c1117' }}
          onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
          preserveAspectRatio="xMidYMid meet"
        >
          <Defs />

          {/* Terrain base (optional) */}
          {layers.terrain && terrain.map((t, i) => <rect key={i} x={t.x} y={t.y} width={t.w} height={t.h} fill={t.fill} opacity={0.7} />)}

          {/* State fill — subtle dark blue-gray with soft border */}
          <polygon points={outPts} fill="#131c28" fillOpacity={layers.terrain ? 0.3 : 0.9} stroke="#2a4060" strokeWidth={0.8} strokeOpacity={0.5} />

          {/* Graticule lines */}
          {[0.2, 0.4, 0.6, 0.8].map(f => (
            <g key={f} opacity={0.08}>
              <line x1={PAD} y1={PAD + iH * f} x2={PAD + iW} y2={PAD + iH * f} stroke="#5588aa" strokeWidth={0.4} />
              <line x1={PAD + iW * f} y1={PAD} x2={PAD + iW * f} y2={PAD + iH} stroke="#5588aa" strokeWidth={0.4} />
            </g>
          ))}

          {/* Cities */}
          {cityPts.map(({ name, cx, cy }) => (
            <g key={name} opacity={0.6}>
              <circle cx={cx} cy={cy} r={1.8} fill="#7aaccc" />
              <text x={cx + 5} y={cy - 3} fontSize="6.5" fontFamily="'DM Sans',sans-serif" fill="#7aaccc" letterSpacing="0.3"
                style={{ pointerEvents: 'none', userSelect: 'none' }}>{name}</text>
            </g>
          ))}

          {/* ── Spread layer ── */}
          {layers.spread && peris.map(p => (
            <polygon key={p.id} points={p.pts} fill="rgba(251,146,60,0.08)" stroke="#fb923c" strokeWidth={1.4} strokeDasharray="8,4" opacity={0.8} />
          ))}
          {layers.spread && spArrows.map(a => (
            <g key={a.k} opacity={0.75}>
              <path d={a.line} stroke="#fb923c" strokeWidth={1.6} strokeDasharray="5,4" fill="none">
                <animate attributeName="stroke-dashoffset" values="0;-18" dur="1.5s" repeatCount="indefinite" />
              </path>
              <path d={a.head} fill="#fb923c" />
            </g>
          ))}
          {layers.spread && proj.map(c => {
            if (srcSet.has(c.grid_id)) return <circle key={`ss${c.grid_id}`} cx={c.cx} cy={c.cy} r={hexR + 7} fill="rgba(239,68,68,0.08)" stroke="#ef4444" strokeWidth={1} strokeDasharray="3,2" opacity={0.5} />;
            if (affSet.has(c.grid_id)) return <circle key={`sa${c.grid_id}`} cx={c.cx} cy={c.cy} r={hexR + 5} fill="rgba(251,146,60,0.06)" stroke="#fb923c" strokeWidth={0.5} opacity={0.4} />;
            return null;
          })}

          {/* ── Hex cells (batched by color, with shadow) ── */}
          {Object.entries(hexBatches).map(([color, paths]) => (
            <path key={color} d={paths.join(' ')} fill={color} fillOpacity={0.88}
              stroke="rgba(0,0,0,0.5)" strokeWidth={0.7} filter="url(#hex-lift)" />
          ))}

          {/* Hover highlight */}
          {hov && hov !== selectedId && proj.filter(c => c.grid_id === hov).map(c => (
            <polygon key="hov" points={hexPoints(c.cx, c.cy, hexR + 2)} fill="none" stroke="rgba(255,255,255,0.45)" strokeWidth={1.5} />
          ))}

          {/* Selection ring */}
          {selectedId && proj.filter(c => c.grid_id === selectedId).map(c => (
            <g key="sel" filter="url(#sel-glow)">
              <polygon points={hexPoints(c.cx, c.cy, hexR + 3)} fill="none" stroke="#60a5fa" strokeWidth={2.5}>
                <animate attributeName="strokeOpacity" values="1;0.5;1" dur="2s" repeatCount="indefinite" />
              </polygon>
            </g>
          ))}

          {/* Active fire glow + pulse */}
          {proj.filter(c => c.fire_detected_binary === 1).map(c => (
            <g key={`fg${c.grid_id}`}>
              <circle cx={c.cx} cy={c.cy} r={hexR + 4} fill="rgba(255,60,0,0.12)" filter="url(#fire-glow)" />
              <circle cx={c.cx} cy={c.cy} r={hexR + 2} fill="none" stroke="#ff4400" strokeWidth={1.2}>
                <animate attributeName="r" values={`${hexR};${hexR + 9};${hexR}`} dur="2.2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.6;0.08;0.6" dur="2.2s" repeatCount="indefinite" />
              </circle>
            </g>
          ))}

          {/* Fire count */}
          {proj.filter(c => c.active_fire_count > 0).map(c => (
            <text key={`n${c.grid_id}`} x={c.cx} y={c.cy + 1} textAnchor="middle" dominantBaseline="middle"
              fontSize="7" fontFamily="'JetBrains Mono',monospace" fill="#fff" fontWeight="bold" style={{ pointerEvents: 'none' }}>
              {c.active_fire_count}
            </text>
          ))}

          {/* Wind arrows */}
          {layers.wind && proj.map(c => {
            const m = edits[c.grid_id] ? { ...c, ...edits[c.grid_id] } : c;
            if (m.wind_direction_10m == null) return null;
            const a = windArrowD(c.cx, c.cy, m.wind_direction_10m, m.wind_speed_10m, zoom < 0.55 ? 1.3 : 1);
            return (
              <g key={`w${c.grid_id}`} opacity={windSpeedOpacity(m.wind_speed_10m)} style={{ pointerEvents: 'none' }}>
                <path d={a.shaft} stroke="#e0f2fe" strokeWidth={1.8} strokeLinecap="round" fill="none" />
                <path d={a.head} fill="#e0f2fe" />
              </g>
            );
          })}

          {/* Crown badges */}
          {layers.crown && proj.map(c => {
            const m = edits[c.grid_id] ? { ...c, ...edits[c.grid_id] } : c;
            const cr = deriveCrownFire(m.canopy_cover_pct, m.canopy_base_height_m, m.canopy_bulk_density);
            if (cr === 'none') return null;
            const cfg = CROWN_CFG[cr]; const bx = c.cx + hexR * .65, by = c.cy - hexR * .65;
            return <g key={`cr${c.grid_id}`} style={{ pointerEvents: 'none' }}><circle cx={bx} cy={by} r={3.5} fill="#111" stroke={cfg.color} strokeWidth={.8} /><text x={bx} y={by + .8} textAnchor="middle" dominantBaseline="middle" fontSize="4.5" fill={cfg.color} fontWeight="bold">{cr === 'active' ? 'A' : 'P'}</text></g>;
          })}

          {/* Note / edit badges */}
          {proj.filter(c => c.notes || edits[c.grid_id]?.notes).map(c => (
            <circle key={`ni${c.grid_id}`} cx={c.cx - hexR * .7} cy={c.cy - hexR * .7} r={2.2} fill="#3b82f6" style={{ pointerEvents: 'none' }} />
          ))}
          {proj.filter(c => edits[c.grid_id]).map(c => (
            <circle key={`ei${c.grid_id}`} cx={c.cx + hexR * .7} cy={c.cy + hexR * .55} r={2.2} fill="#a855f7" style={{ pointerEvents: 'none' }} />
          ))}

          {/* Labels at high zoom */}
          {showLabels && proj.map(c => (
            <text key={`l${c.grid_id}`} x={c.cx} y={c.cy + hexR + 7} textAnchor="middle"
              fontSize="5" fontFamily="'DM Sans',sans-serif" fill="#8ab0cc" opacity={0.55} style={{ pointerEvents: 'none', userSelect: 'none' }}>
              {c.name.length > 16 ? c.name.slice(0, 14) + '..' : c.name}
            </text>
          ))}

          {/* Compass */}
          <g opacity={0.35}>
            <text x={PAD + 3} y={PAD + 9} fontSize="7" fontFamily="'DM Sans',sans-serif" fill="#5588aa">{label.toUpperCase()}</text>
            <text x={MAP_W - PAD - 6} y={PAD + 9} fontSize="7" fontFamily="'DM Sans',sans-serif" fill="#5588aa" textAnchor="middle">N</text>
            <line x1={MAP_W - PAD - 6} y1={PAD + 11} x2={MAP_W - PAD - 6} y2={PAD + 18} stroke="#5588aa" strokeWidth={.8} />
            <polygon points={`${MAP_W-PAD-6},${PAD+10} ${MAP_W-PAD-9},${PAD+14} ${MAP_W-PAD-3},${PAD+14}`} fill="#5588aa" />
          </g>
        </svg>

        {/* Hover tooltip */}
        {hovCell && hovCell.grid_id !== selectedId && (
          <div className="absolute top-3 left-3 px-2.5 py-1.5 bg-surface-1/95 backdrop-blur border border-border-subtle rounded-lg shadow-xl pointer-events-none z-10">
            <div className="text-[10px] font-semibold text-text-primary">{hovCell.name}</div>
            <div className="text-[9px] font-mono text-text-muted mt-0.5">
              {(hovCell.fire_risk_score * 100).toFixed(0)}% — {getRiskTier(hovCell.fire_risk_score)}
              {hovCell.fire_detected_binary === 1 && <span className="text-risk-critical font-semibold ml-1.5">FIRE</span>}
            </div>
          </div>
        )}

        <ZoomBtns onIn={zIn} onOut={zOut} onFit={fit} pct={Math.round(1 / zoom * 100)} />
      </div>
    </div>
  );
});

// ─── Edit form ────────────────────────────────────────────────────────────────
function EditForm({ cell, vals, onChange, onSave, onCancel }) {
  const fields = [
    { k: 'temperature_2m', l: 'Temp', u: '°C', s: .1 }, { k: 'relative_humidity_2m', l: 'RH', u: '%', s: .1, mn: 0, mx: 100 },
    { k: 'wind_speed_10m', l: 'Wind Spd', u: 'm/s', s: .1, mn: 0 }, { k: 'wind_direction_10m', l: 'Wind Dir', u: '°', s: 1, mn: 0, mx: 360 },
    { k: 'precipitation', l: 'Precip', u: 'mm', s: .1, mn: 0 }, { k: 'soil_moisture_0_to_7cm', l: 'Soil Moist', u: 'm³/m³', s: .01, mn: 0, mx: .5 },
    { k: 'active_fire_count', l: 'Fire Count', u: '', s: 1, mn: 0 },
  ];
  return (
    <div className="px-4 py-3 border-b border-border-subtle bg-accent-blue/5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-mono font-semibold text-accent-blue uppercase tracking-wider">Manual Override</span>
        <button onClick={onCancel} className="text-text-muted hover:text-text-primary"><X className="w-3 h-3" /></button>
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        {fields.map(f => (
          <label key={f.k} className="flex flex-col gap-0.5">
            <span className="text-[9px] font-mono text-text-muted">{f.l} <span className="opacity-40">{f.u}</span></span>
            <input type="number" step={f.s} min={f.mn} max={f.mx} value={vals[f.k] ?? cell[f.k] ?? ''}
              onChange={e => onChange(f.k, e.target.value === '' ? undefined : +e.target.value)}
              className="w-full bg-surface-2 border border-border-subtle rounded px-1.5 py-1 text-[10px] font-mono text-text-primary focus:border-accent-blue/40 focus:outline-none" />
          </label>
        ))}
      </div>
      <label className="flex flex-col gap-0.5 mt-2">
        <span className="text-[9px] font-mono text-text-muted">Notes</span>
        <textarea value={vals.notes ?? cell.notes ?? ''} onChange={e => onChange('notes', e.target.value || undefined)} rows={2}
          className="w-full bg-surface-2 border border-border-subtle rounded px-1.5 py-1 text-[10px] font-mono text-text-primary focus:border-accent-blue/40 focus:outline-none resize-none" />
      </label>
      <div className="flex gap-2 mt-2">
        <button onClick={onSave} className="flex-1 flex items-center justify-center gap-1 py-1.5 bg-accent-blue/15 border border-accent-blue/30 rounded text-accent-blue text-[10px] font-mono font-semibold hover:bg-accent-blue/25 transition-colors"><Save className="w-3 h-3" />Save</button>
        <button onClick={onCancel} className="flex-1 py-1.5 bg-surface-2 border border-border-subtle rounded text-text-muted text-[10px] font-mono hover:text-text-primary transition-colors">Cancel</button>
      </div>
    </div>
  );
}

// ─── Detail Panel ─────────────────────────────────────────────────────────────
function DetailPanel({ cellId, allCells, edits, setEdits, onNavigate }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({});
  const cell = allCells.find(c => c.grid_id === cellId);
  const merged = cell ? { ...cell, ...(edits[cellId] || {}) } : null;
  const spread = cellId ? OBJ2_SPREAD[cellId] : null;
  const tier = merged ? getRiskTier(merged.fire_risk_score) : null;
  const crown = merged ? deriveCrownFire(merged.canopy_cover_pct, merged.canopy_base_height_m, merged.canopy_bulk_density) : 'none';
  const startEdit = () => { setDraft(edits[cellId] || {}); setEditing(true); };
  const cancelEdit = () => { setDraft({}); setEditing(false); };
  const saveEdit = () => {
    const ne = { ...edits }; const cl = Object.fromEntries(Object.entries(draft).filter(([, v]) => v !== undefined));
    if (Object.keys(cl).length) ne[cellId] = cl; else delete ne[cellId];
    setEdits(ne); localStorage.setItem('fireMapEdits', JSON.stringify(ne)); setEditing(false);
  };
  const exportJSON = () => {
    if (!merged) return;
    const p = { region: merged.lat > 36 ? 'california' : 'texas', grid_ids: [merged.grid_id],
      features: { temperature_2m: [merged.temperature_2m], relative_humidity_2m: [merged.relative_humidity_2m], wind_speed_10m: [merged.wind_speed_10m], wind_direction_10m: [merged.wind_direction_10m], precipitation: [merged.precipitation], soil_moisture_0_to_7cm: [merged.soil_moisture_0_to_7cm], vpd: [merged.vpd], elevation_m: [merged.elevation_m], slope_degrees: [merged.slope_degrees], aspect_degrees: [merged.aspect_degrees], canopy_cover_pct: [merged.canopy_cover_pct], canopy_base_height_m: [merged.canopy_base_height_m], canopy_bulk_density: [merged.canopy_bulk_density], fuel_model_fbfm40: [merged.fuel_model_fbfm40] },
      _meta: { exported_at: new Date().toISOString(), source: 'fire-map-manual-override' } };
    const b = new Blob([JSON.stringify(p, null, 2)], { type: 'application/json' }); const u = URL.createObjectURL(b);
    const a = document.createElement('a'); a.href = u; a.download = `predict_${merged.grid_id.slice(0, 8)}.json`; a.click(); URL.revokeObjectURL(u);
  };

  if (!cell) return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-text-muted px-6">
      <Crosshair className="w-10 h-10 opacity-15" />
      <span className="text-[11px] text-center leading-relaxed opacity-50">Click a cell on the map<br/>to inspect details</span>
    </div>
  );

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className={`px-4 py-3 border-b border-border-subtle ${tier === 'CRITICAL' ? 'glow-critical bg-risk-critical/5' : ''}`}>
        <div className="flex items-start justify-between gap-2 mb-1">
          <div><div className="text-text-primary text-[13px] font-semibold leading-tight">{merged.name}</div><div className="text-text-muted text-[9px] font-mono mt-0.5">{merged.grid_id.slice(0, 12)}...</div></div>
          <RiskBadge tier={tier} />
        </div>
        <div className="text-[10px] font-mono text-text-muted">{merged.lat.toFixed(2)}°N, {Math.abs(merged.lon).toFixed(2)}°W</div>
        {edits[cellId] && <div className="mt-1 flex items-center gap-1 text-[9px] font-mono text-purple-400"><Pencil className="w-2.5 h-2.5" />Overrides active</div>}
      </div>

      {editing ? <EditForm cell={cell} vals={draft} onChange={(k, v) => setDraft(d => ({ ...d, [k]: v }))} onSave={saveEdit} onCancel={cancelEdit} /> : <>
        {/* Risk */}
        <div className="px-4 py-3 border-b border-border-subtle">
          <div className="flex items-center gap-1.5 mb-2"><Activity className="w-3 h-3 text-accent-blue" /><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Ignition Risk</span></div>
          <div className="flex justify-between items-center mb-1"><span className="text-[10px] font-mono text-text-muted">P(ignition)</span><span className={`text-[14px] font-mono font-bold ${TIER_COLORS[tier].text}`}>{(merged.fire_risk_score * 100).toFixed(1)}%</span></div>
          <div className="h-2 bg-surface-3 rounded-full overflow-hidden"><div className="h-full rounded-full transition-all duration-500" style={{ width: `${merged.fire_risk_score * 100}%`, background: `linear-gradient(90deg, ${TIER_COLORS.LOW.fill}, ${TIER_COLORS[tier].fill})`, boxShadow: `0 0 8px ${TIER_COLORS[tier].glow}` }} /></div>
          <div className="flex gap-2 mt-1.5">{[['CRIT','65%'],['HIGH','36.5%'],['MED','15%']].map(([l,v])=><span key={l} className="text-[9px] font-mono text-text-muted">{l}:{v}</span>)}</div>
        </div>

        {/* Weather */}
        <div className="px-4 py-3 border-b border-border-subtle">
          <div className="flex items-center gap-1.5 mb-2"><Thermometer className="w-3 h-3 text-accent-orange" /><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Weather</span></div>
          <div className="divide-y divide-border-subtle/30">
            <Row icon={Thermometer} label="Temperature" value={merged.temperature_2m} unit="°C" />
            <Row icon={Droplets} label="Humidity" value={merged.relative_humidity_2m} unit="%" />
            <Row icon={Wind} label="Wind" value={`${merged.wind_speed_10m} m/s ${compassLabel(merged.wind_direction_10m)}`} />
            <Row icon={Activity} label="VPD" value={merged.vpd} unit="kPa" />
            <Row icon={Droplets} label="Soil Moisture" value={merged.soil_moisture_0_to_7cm} unit="m³/m³" />
          </div>
        </div>

        {/* Terrain */}
        <div className="px-4 py-3 border-b border-border-subtle">
          <div className="flex items-center gap-1.5 mb-2"><TreePine className="w-3 h-3 text-green-500" /><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Terrain & Canopy</span></div>
          <div className="divide-y divide-border-subtle/30">
            <Row icon={Mountain} label="Elevation" value={merged.elevation_m} unit="m" />
            <Row icon={Activity} label="Slope" value={merged.slope_degrees} unit="°" />
            <Row icon={TreePine} label="Canopy Cover" value={merged.canopy_cover_pct} unit="%" />
            <Row icon={TreePine} label="Base Height" value={merged.canopy_base_height_m} unit="m" />
            <Row icon={Layers} label="Fuel Model" value={merged.fuel_model_fbfm40} />
            <Row icon={TreePine} label="Vegetation" value={merged.vegetation_type} />
          </div>
          {crown !== 'none' && (
            <div className={`mt-2 px-2.5 py-2 rounded-lg border ${crown === 'active' ? 'bg-risk-critical/8 border-risk-critical/25' : 'bg-risk-high/8 border-risk-high/25'}`}>
              <span className={`text-[10px] font-mono font-semibold ${crown === 'active' ? 'text-risk-critical' : 'text-risk-high'}`}>Crown Fire: {crown.toUpperCase()}</span>
              <div className="text-[9px] font-mono text-text-muted mt-0.5">{crown === 'active' ? 'Sustained crown fire likely — high bulk density, low base height' : 'Passive torching possible — low canopy base'}</div>
            </div>
          )}
        </div>

        {/* Active fire */}
        {merged.fire_detected_binary === 1 && (
          <div className="mx-4 my-2 px-3 py-2.5 bg-risk-critical/8 border border-risk-critical/25 rounded-lg glow-critical">
            <div className="flex items-center gap-2"><Flame className="w-3.5 h-3.5 text-risk-critical" /><span className="text-[10px] font-mono font-bold text-risk-critical">ACTIVE FIRE — {merged.active_fire_count} hotspots</span></div>
            {merged.mean_frp > 0 && <div className="text-[9px] font-mono text-text-muted mt-1">FRP: {merged.mean_frp} MW · {merged.mean_frp > 200 ? 'High' : merged.mean_frp > 50 ? 'Moderate' : 'Low'} intensity</div>}
          </div>
        )}

        {/* Spread */}
        {spread && (
          <div className="px-4 py-3 border-b border-border-subtle">
            <div className="flex items-center gap-1.5 mb-2"><Navigation className="w-3 h-3 text-accent-orange" /><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Spread Simulation</span></div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
              {[['Rate',`${spread.spread_rate_m_per_min} m/min`],['Area',`${spread.spread_area_km2} km²`],['Horizon',`${spread.time_horizon_hrs}h`],['Contain.',`${(spread.containment_probability*100).toFixed(0)}%`],['Conf.',`${(spread.confidence*100).toFixed(0)}%`],['Wind',`${spread.wind_speed_m_s} m/s ${compassLabel(spread.wind_direction_deg)}`]].map(([k,v])=><div key={k} className="flex justify-between"><span className="text-[10px] font-mono text-text-muted">{k}</span><span className="text-[10px] font-mono text-text-primary">{v}</span></div>)}
            </div>
            {spread.notes && <div className="mt-2 text-[9px] font-mono text-text-muted leading-relaxed">{spread.notes}</div>}
          </div>
        )}

        {/* Notes */}
        {merged.notes && (
          <div className="px-4 py-3 border-b border-border-subtle">
            <div className="flex items-center gap-1.5 mb-1"><Info className="w-3 h-3 text-accent-blue" /><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Notes</span></div>
            <div className="text-[10px] font-mono text-text-muted leading-relaxed">{merged.notes}</div>
          </div>
        )}

        {/* Actions */}
        <div className="px-4 py-3 space-y-2">
          <button onClick={startEdit} className="w-full flex items-center justify-between px-3 py-2.5 bg-accent-blue/8 border border-accent-blue/25 rounded-lg text-accent-blue hover:bg-accent-blue/15 transition-colors">
            <span className="flex items-center gap-1.5"><Pencil className="w-3 h-3" /><span className="text-[10px] font-mono font-semibold">Edit Features</span></span><ChevronRight className="w-3 h-3" />
          </button>
          <button onClick={exportJSON} className="w-full flex items-center justify-between px-3 py-2.5 bg-surface-2 border border-border-subtle rounded-lg text-text-secondary hover:text-text-primary hover:bg-surface-3 transition-colors">
            <span className="flex items-center gap-1.5"><Download className="w-3 h-3" /><span className="text-[10px] font-mono">Export JSON</span></span><ChevronRight className="w-3 h-3" />
          </button>
          {edits[cellId] && <button onClick={() => { const n={...edits}; delete n[cellId]; setEdits(n); localStorage.setItem('fireMapEdits',JSON.stringify(n)); }} className="w-full flex items-center justify-center gap-1.5 px-3 py-2 text-text-muted hover:text-risk-critical text-[10px] font-mono transition-colors"><X className="w-3 h-3" />Reset Overrides</button>}
          {spread && <button onClick={() => onNavigate?.('reports')} className="w-full flex items-center justify-between px-3 py-2.5 bg-surface-2 border border-border-subtle rounded-lg text-text-secondary hover:text-text-primary hover:bg-surface-3 transition-colors"><span className="flex items-center gap-1.5"><FileText className="w-3 h-3" /><span className="text-[10px] font-mono">Incident Report</span></span><ChevronRight className="w-3 h-3" /></button>}
        </div>
      </>}
    </div>
  );
}

// ─── Legend ────────────────────────────────────────────────────────────────────
function Legend({ layers }) {
  return (
    <div className="flex items-center gap-4 px-4 py-1.5 bg-surface-1 border-t border-border-subtle flex-shrink-0 overflow-x-auto">
      <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider shrink-0">Legend</span>
      {layers.colorLayer === 'risk' && <div className="flex items-center gap-2.5">{['CRITICAL','HIGH','MEDIUM','LOW'].map(t=><div key={t} className="flex items-center gap-1"><div className="w-2.5 h-2.5 rounded-sm shadow-sm" style={{ background: TIER_COLORS[t].fill }} /><span className="text-[9px] font-mono text-text-muted">{t}</span></div>)}</div>}
      {layers.colorLayer === 'moisture' && <div className="flex items-center gap-1"><div className="w-16 h-2 rounded" style={{ background: 'linear-gradient(90deg,#78350f,#b45309,#d97706,#a3e635,#22c55e,#047857)' }} /><span className="text-[9px] font-mono text-text-muted">Dry→Wet</span></div>}
      {layers.colorLayer === 'intensity' && <div className="flex items-center gap-1"><div className="w-16 h-2 rounded" style={{ background: 'linear-gradient(90deg,#422006,#fbbf24,#f97316,#ef4444,#7c3aed)' }} /><span className="text-[9px] font-mono text-text-muted">FRP</span></div>}
      <div className="flex items-center gap-1"><div className="w-2.5 h-2.5 rounded-full border-2 border-red-500 animate-pulse" /><span className="text-[9px] font-mono text-text-muted">Fire</span></div>
      {layers.wind && <div className="flex items-center gap-1"><Wind className="w-3 h-3 text-sky-100/70" /><span className="text-[9px] font-mono text-text-muted">Wind</span></div>}
      {layers.spread && <div className="flex items-center gap-1"><div className="w-5 h-1" style={{ borderBottom: '2px dashed #fb923c' }} /><span className="text-[9px] font-mono text-text-muted">Spread</span></div>}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function FireMap({ onNavigate }) {
  const [layers, setLayers] = useState({ colorLayer: 'risk', wind: true, crown: false, spread: false, terrain: false });
  const [sel, setSel] = useState(null);
  const [res, setRes] = useState('64km');
  const [edits, setEdits] = useState(() => { try { return JSON.parse(localStorage.getItem('fireMapEdits') || '{}'); } catch { return {}; } });

  const ca = useMemo(() => res === '22km' ? generateFireZoomCells(CALIFORNIA_CELLS_ENRICHED, OBJ2_SPREAD) : CALIFORNIA_CELLS_ENRICHED, [res]);
  const tx = useMemo(() => res === '22km' ? generateFireZoomCells(TEXAS_CELLS_ENRICHED, OBJ2_SPREAD) : TEXAS_CELLS_ENRICHED, [res]);
  const all = useMemo(() => [...ca.map(c => ({ ...c, region: 'california' })), ...tx.map(c => ({ ...c, region: 'texas' }))], [ca, tx]);
  const stats = useMemo(() => ({ critical: all.filter(c => getRiskTier(c.fire_risk_score) === 'CRITICAL').length, high: all.filter(c => getRiskTier(c.fire_risk_score) === 'HIGH').length, activeFires: all.filter(c => c.fire_detected_binary === 1).length, total: all.length }), [all]);

  return (
    <div className="flex h-full overflow-hidden bg-surface-0">
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <LayerBar layers={layers} setLayers={setLayers} stats={stats} resolution={res} setResolution={setRes} />
        <div className="flex-1 flex overflow-hidden p-3 gap-3">
          <RegionMap cells={ca} bounds={CA_BOUNDS} label="California" layers={layers} selectedId={sel} spreadData={OBJ2_SPREAD} onSelect={useCallback(id => setSel(p => p === id ? null : id), [])} edits={edits} />
          <RegionMap cells={tx} bounds={TX_BOUNDS} label="Texas" layers={layers} selectedId={sel} spreadData={OBJ2_SPREAD} onSelect={useCallback(id => setSel(p => p === id ? null : id), [])} edits={edits} />
        </div>
        <Legend layers={layers} />
      </div>
      <div className="w-[280px] flex-shrink-0 border-l border-border-subtle bg-surface-1 flex flex-col">
        <div className="px-4 py-2.5 border-b border-border-subtle flex items-center gap-2 flex-shrink-0">
          <MapPin className="w-3.5 h-3.5 text-text-muted" /><span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">Cell Detail</span>
          {sel && <button onClick={() => setSel(null)} className="ml-auto text-[9px] font-mono text-text-muted hover:text-text-primary">clear</button>}
        </div>
        <div className="flex-1 overflow-hidden"><DetailPanel cellId={sel} allCells={all} edits={edits} setEdits={setEdits} onNavigate={onNavigate} /></div>
      </div>
    </div>
  );
}
