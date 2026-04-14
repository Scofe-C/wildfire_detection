import { useState, useMemo } from 'react';
import {
  Layers, Flame, Wind, Thermometer, Droplets, Activity,
  AlertTriangle, FileText, ChevronRight, Info, Crosshair, MapPin,
} from 'lucide-react';
import { CALIFORNIA_CELLS, TEXAS_CELLS, getRiskTier } from '../../data/mockGridData';
import { OBJ1_PREDICTIONS, OBJ2_SPREAD, MAP_META } from '../../data/mockMapData';

// ─── Color helpers ────────────────────────────────────────────────────────────
const TIER_CFG = {
  CRITICAL: { fill: '#ff3333', stroke: '#ff3333', glow: 'rgba(255,51,51,0.5)',  label: 'CRITICAL', text: 'text-risk-critical' },
  HIGH:     { fill: '#ff6b00', stroke: '#ff6b00', glow: 'rgba(255,107,0,0.4)',  label: 'HIGH',     text: 'text-risk-high'     },
  MEDIUM:   { fill: '#fbbf24', stroke: '#fbbf24', glow: 'rgba(251,191,36,0.3)', label: 'MEDIUM',   text: 'text-risk-medium'   },
  LOW:      { fill: '#00e676', stroke: '#00e676', glow: 'rgba(0,230,118,0.25)', label: 'LOW',      text: 'text-risk-low'      },
};

const SPREAD_FILL  = 'rgba(255,107,0,0.18)';
const SPREAD_STR   = '#ff6b00';
const SOURCE_FILL  = 'rgba(255,51,51,0.25)';

// ─── Map projection ───────────────────────────────────────────────────────────
// Simple linear lat/lon → SVG pixel projection for each region panel.
const CA_BOUNDS = { minLat: 32.2, maxLat: 42.2, minLon: -124.8, maxLon: -113.8 };
const TX_BOUNDS = { minLat: 25.4, maxLat: 37.0, minLon: -107.2, maxLon: -92.8  };

function project(lat, lon, bounds, w, h) {
  const x = ((lon - bounds.minLon) / (bounds.maxLon - bounds.minLon)) * w;
  const y = ((bounds.maxLat - lat) / (bounds.maxLat - bounds.minLat)) * h;
  return { x, y };
}

// ─── Simplified state outlines ────────────────────────────────────────────────
// Approximate [lat, lon] boundary polygons for geographic context only.
// Not surveyed data — sufficient for visual orientation at 64km grid resolution.
const CA_OUTLINE = [
  [42.0,-124.2],[41.4,-124.1],[40.4,-124.4],[39.8,-123.8],
  [38.8,-123.9],[38.3,-123.1],[37.9,-122.7],[37.5,-122.4],
  [37.2,-122.2],[36.6,-121.9],[35.7,-121.3],[35.0,-120.9],
  [34.5,-120.5],[34.1,-119.5],[33.8,-118.5],[33.4,-117.9],
  [32.7,-117.2],[32.5,-117.1],[32.5,-114.7],[33.0,-114.6],
  [34.0,-114.6],[35.2,-114.6],[36.0,-114.7],[37.0,-114.0],
  [38.0,-114.1],[39.0,-114.1],[40.0,-114.1],[41.0,-114.0],
  [42.0,-114.1],[42.0,-124.2],
];
const TX_OUTLINE = [
  [36.5,-103.0],[36.5,-100.0],[34.0,-100.0],[33.8,-96.5],
  [33.4,-94.0],[31.0,-94.0],[30.0,-93.8],[29.5,-93.9],
  [28.7,-95.7],[27.8,-97.4],[26.2,-97.3],[26.5,-99.1],
  [28.3,-100.3],[29.2,-101.2],[29.8,-103.3],[30.0,-104.6],
  [31.0,-105.3],[31.8,-106.5],[32.0,-106.6],[32.0,-103.0],
  [36.5,-103.0],
];

// Reference cities for geographic orientation
const CA_CITIES = [
  { name: 'Los Angeles',   lat: 34.05, lon: -118.24 },
  { name: 'San Francisco', lat: 37.77, lon: -122.42 },
  { name: 'Santa Barbara', lat: 34.42, lon: -119.70 },
];
const TX_CITIES = [
  { name: 'Austin',  lat: 30.27, lon: -97.74  },
  { name: 'Dallas',  lat: 32.78, lon: -96.80  },
  { name: 'El Paso', lat: 31.78, lon: -106.40 },
];

// ─── Sub-components ───────────────────────────────────────────────────────────
function RiskBadge({ tier }) {
  const cfg = TIER_CFG[tier] || TIER_CFG.LOW;
  return (
    <span className={`text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded border leading-none
      ${tier === 'CRITICAL' ? 'bg-risk-critical/20 text-risk-critical border-risk-critical/40' :
        tier === 'HIGH'     ? 'bg-risk-high/20 text-risk-high border-risk-high/40' :
        tier === 'MEDIUM'   ? 'bg-risk-medium/20 text-risk-medium border-risk-medium/40' :
                              'bg-risk-low/20 text-risk-low border-risk-low/40'}`}>
      {tier}
    </span>
  );
}

function FeatureRow({ icon: Icon, label, value, unit }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <div className="flex items-center gap-1.5 text-text-muted">
        <Icon className="w-3 h-3" />
        <span className="text-[10px] font-mono">{label}</span>
      </div>
      <span className="text-[10px] font-mono text-text-primary">{value} {unit}</span>
    </div>
  );
}

// ─── Map panel (SVG) ──────────────────────────────────────────────────────────
function RegionMap({ cells, bounds, label, activeLayer, selectedId, spreadData, onSelect, w = 300, h = 320 }) {
  const PAD = 20;
  const mapW = w - PAD * 2;
  const mapH = h - PAD * 2;

  // Geographic context: derive outline + cities from label prop
  const outline = label === 'California' ? CA_OUTLINE : TX_OUTLINE;
  const cities  = label === 'California' ? CA_CITIES  : TX_CITIES;

  // Project state outline boundary to SVG pixel coords
  const outlinePts = useMemo(() =>
    outline.map(([lat, lon]) => {
      const { x, y } = project(lat, lon, bounds, mapW, mapH);
      return `${x + PAD},${y + PAD}`;
    }).join(' ')
  , [outline, bounds, mapW, mapH]);

  // Project city reference positions
  const cityDots = useMemo(() =>
    cities.map(({ name, lat, lon }) => {
      const { x, y } = project(lat, lon, bounds, mapW, mapH);
      return { name, cx: x + PAD, cy: y + PAD };
    })
  , [cities, bounds, mapW, mapH]);

  // Determine which cells are in the current spread footprint
  const spreadCells = useMemo(() => {
    if (activeLayer !== 'spread') return new Set();
    const affected = new Set();
    Object.values(spreadData).forEach(sim => {
      sim.affected_cells.forEach(id => affected.add(id));
    });
    return affected;
  }, [activeLayer, spreadData]);

  const sourceCells = useMemo(() => {
    if (activeLayer !== 'spread') return new Set();
    return new Set(Object.keys(spreadData));
  }, [activeLayer, spreadData]);

  // Build spread perimeter polygons for cells in this region
  const perimeterPolys = useMemo(() => {
    if (activeLayer !== 'spread') return [];
    return Object.values(spreadData)
      .filter(sim => cells.some(c => c.grid_id === sim.source_cell))
      .map(sim => ({
        id: sim.source_cell,
        pts: sim.perimeter_coords
          .map(({ lat, lon }) => {
            const { x, y } = project(lat, lon, bounds, mapW, mapH);
            return `${x + PAD},${y + PAD}`;
          })
          .join(' '),
        confidence: sim.confidence,
      }));
  }, [activeLayer, spreadData, cells, bounds, mapW, mapH]);

  return (
    <div className="flex flex-col items-center">
      <div className="text-[9px] font-mono text-text-muted uppercase tracking-widest mb-1">{label}</div>
      <svg
        width={w}
        height={h}
        className="bg-surface-2 rounded border border-border-subtle"
        style={{ cursor: 'crosshair' }}
      >
        {/* State boundary fill — subtle geographic context, rendered first */}
        <polygon
          points={outlinePts}
          fill="#1a3a5c"
          fillOpacity={0.14}
          stroke="#6b8fa8"
          strokeWidth={0.7}
          strokeOpacity={0.3}
        />

        {/* Grid reference lines */}
        {[0.25, 0.5, 0.75].map(f => (
          <g key={f}>
            <line x1={PAD} y1={PAD + mapH * f} x2={PAD + mapW} y2={PAD + mapH * f}
              stroke="#1e2d3d" strokeWidth={0.5} />
            <line x1={PAD + mapW * f} y1={PAD} x2={PAD + mapW * f} y2={PAD + mapH}
              stroke="#1e2d3d" strokeWidth={0.5} />
          </g>
        ))}

        {/* City reference labels — geographic orientation, behind hex cells */}
        {cityDots.map(({ name, cx, cy }) => (
          <g key={name}>
            <circle cx={cx} cy={cy} r={1.8} fill="#4a6a80" opacity={0.5} />
            <text
              x={cx + 4} y={cy - 2}
              fontSize="7" fontFamily="monospace"
              fill="#4a6a80" opacity={0.6}
              style={{ pointerEvents: 'none', userSelect: 'none' }}
            >
              {name}
            </text>
          </g>
        ))}

        {/* Spread perimeter polygons */}
        {perimeterPolys.map(poly => (
          <polygon
            key={poly.id}
            points={poly.pts}
            fill={SOURCE_FILL}
            stroke={SPREAD_STR}
            strokeWidth={1.2}
            strokeDasharray="4,2"
            opacity={0.85}
          />
        ))}

        {/* Cells */}
        {cells.map(cell => {
          const pred = OBJ1_PREDICTIONS[cell.grid_id];
          const tier = pred ? pred.tier : getRiskTier(cell.fire_risk_score);
          const cfg  = TIER_CFG[tier];
          const { x, y } = project(cell.lat, cell.lon, bounds, mapW, mapH);
          const cx = x + PAD;
          const cy = y + PAD;
          const isSelected = cell.grid_id === selectedId;
          const isSource   = sourceCells.has(cell.grid_id);
          const isSpread   = spreadCells.has(cell.grid_id) && !isSource;
          const isActive   = cell.fire_detected_binary === 1;

          // Hexagon shape via clip/polygon approximation (regular hex)
          const r = 10;
          const hexPts = [0,1,2,3,4,5].map(i => {
            const a = Math.PI / 180 * (60 * i - 30);
            return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
          }).join(' ');

          return (
            <g key={cell.grid_id} onClick={() => onSelect(cell.grid_id)} style={{ cursor: 'pointer' }}>
              {/* Glow ring for critical/selected */}
              {(tier === 'CRITICAL' || isSelected) && (
                <polygon
                  points={hexPts}
                  fill="none"
                  stroke={isSelected ? '#60a5fa' : cfg.stroke}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                  opacity={0.7}
                  transform={`scale(1.35) translate(${cx*(1-1.35)/1.35 / 1 }, ${cy*(1-1.35)/1.35 / 1})`}
                  style={{
                    transformOrigin: `${cx}px ${cy}px`,
                    transform: `translate(0,0) scale(1.35)`,
                  }}
                />
              )}

              {/* Spread highlight */}
              {isSpread && (
                <circle cx={cx} cy={cy} r={r + 5} fill={SPREAD_FILL} stroke={SPREAD_STR} strokeWidth={0.8} opacity={0.6} />
              )}
              {isSource && activeLayer === 'spread' && (
                <circle cx={cx} cy={cy} r={r + 7} fill={SOURCE_FILL} stroke="#ff3333" strokeWidth={1.2} opacity={0.7} />
              )}

              {/* Hex body */}
              <polygon
                points={hexPts}
                fill={cfg.fill}
                fillOpacity={activeLayer === 'spread' && !isSpread && !isSource ? 0.25 : 0.75}
                stroke={isSelected ? '#60a5fa' : cfg.stroke}
                strokeWidth={isSelected ? 2 : 0.8}
              />

              {/* Active fire pulse ring */}
              {isActive && (
                <circle cx={cx} cy={cy} r={r + 3} fill="none" stroke="#ff3333" strokeWidth={1.5} opacity={0.6}>
                  <animate attributeName="r" values={`${r+2};${r+7};${r+2}`} dur="1.8s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.6;0.1;0.6" dur="1.8s" repeatCount="indefinite" />
                </circle>
              )}

              {/* Fire count dot */}
              {cell.active_fire_count > 0 && (
                <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
                  fontSize="7" fontFamily="monospace" fill="#fff" fontWeight="bold">
                  {cell.active_fire_count}
                </text>
              )}
            </g>
          );
        })}

        {/* Region label */}
        <text x={PAD + 4} y={PAD + 10} fontSize="8" fontFamily="monospace" fill="#3d5a73">{label.toUpperCase()}</text>
        {/* N arrow */}
        <text x={w - PAD - 8} y={PAD + 12} fontSize="8" fontFamily="monospace" fill="#3d5a73">N</text>
        <line x1={w - PAD - 5} y1={PAD + 14} x2={w - PAD - 5} y2={PAD + 22} stroke="#3d5a73" strokeWidth={1} />
        <polygon points={`${w-PAD-5},${PAD+13} ${w-PAD-8},${PAD+18} ${w-PAD-2},${PAD+18}`} fill="#3d5a73" />
      </svg>
    </div>
  );
}

// ─── Detail panel ─────────────────────────────────────────────────────────────
function DetailPanel({ cellId, allCells, activeLayer, onNavigate }) {
  const cell   = allCells.find(c => c.grid_id === cellId);
  const pred   = cellId ? OBJ1_PREDICTIONS[cellId] : null;
  const spread = cellId ? OBJ2_SPREAD[cellId] : null;
  const tier   = pred ? pred.tier : (cell ? getRiskTier(cell.fire_risk_score) : null);

  if (!cell) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-text-muted">
        <Crosshair className="w-8 h-8 opacity-30" />
        <span className="text-[11px] font-mono text-center leading-relaxed px-4">
          Click a cell on the map<br />to view details
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Cell header */}
      <div className={`px-4 py-3 border-b border-border-subtle ${tier === 'CRITICAL' ? 'glow-critical bg-risk-critical/5' : ''}`}>
        <div className="flex items-start justify-between gap-2 mb-1">
          <div>
            <div className="text-text-primary text-xs font-semibold leading-tight">{cell.name}</div>
            <div className="text-text-muted text-[10px] font-mono mt-0.5">{cell.grid_id}</div>
          </div>
          <RiskBadge tier={tier} />
        </div>
        <div className="text-[10px] font-mono text-text-muted">
          {cell.lat.toFixed(2)}°N, {Math.abs(cell.lon).toFixed(2)}°W · {cell.region ?? (cell.lat > 36 ? 'California' : cell.lon < -100 ? 'Texas' : 'California')}
        </div>
      </div>

      {/* OBJ-1 Section */}
      <div className="px-4 py-3 border-b border-border-subtle">
        <div className="flex items-center gap-1.5 mb-2">
          <Activity className="w-3 h-3 text-accent-blue" />
          <span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">OBJ-1 Ignition Risk</span>
        </div>

        {pred ? (
          <>
            {/* Probability bar */}
            <div className="mb-2">
              <div className="flex justify-between items-center mb-1">
                <span className="text-[10px] font-mono text-text-muted">P(ignition)</span>
                <span className={`text-sm font-mono font-bold ${TIER_CFG[tier].text}`}>
                  {(pred.probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-surface-3 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${pred.probability * 100}%`,
                    background: TIER_CFG[tier].fill,
                    boxShadow: `0 0 6px ${TIER_CFG[tier].glow}`,
                  }}
                />
              </div>
            </div>

            {/* Threshold markers */}
            <div className="flex gap-2 mb-3">
              {[['CRIT', '65%'], ['HIGH', '36.5%'], ['MED', '15%']].map(([l, v]) => (
                <span key={l} className="text-[9px] font-mono text-text-muted">{l}:{v}</span>
              ))}
            </div>

            {/* Features */}
            <div className="space-y-0 divide-y divide-border-subtle/40">
              <FeatureRow icon={Thermometer} label="Temp"     value={pred.features.temperature_2m}      unit="°C" />
              <FeatureRow icon={Droplets}   label="RH"        value={pred.features.relative_humidity_2m} unit="%"  />
              <FeatureRow icon={Wind}       label="Wind"      value={pred.features.wind_speed_10m}       unit="m/s" />
              <FeatureRow icon={Activity}   label="VPD"       value={pred.features.vpd}                  unit="kPa" />
              <FeatureRow icon={Flame}      label="FWI"       value={pred.features.fire_weather_index}   unit=""   />
            </div>

            <div className="mt-2 text-[9px] font-mono text-text-muted">
              model: {pred.model_version}<br />
              inferred: {pred.inference_ts.replace('T', ' ').slice(0, 19)} UTC
            </div>
          </>
        ) : (
          <div className="text-[10px] font-mono text-text-muted">No prediction available</div>
        )}
      </div>

      {/* Active fire indicator */}
      {cell.fire_detected_binary === 1 && (
        <div className="mx-4 my-2 px-2.5 py-2 bg-risk-critical/10 border border-risk-critical/30 rounded glow-critical">
          <div className="flex items-center gap-1.5">
            <Flame className="w-3 h-3 text-risk-critical" />
            <span className="text-[10px] font-mono font-semibold text-risk-critical">
              ACTIVE FIRE DETECTED — {cell.active_fire_count} thermal anomalies
            </span>
          </div>
        </div>
      )}

      {/* OBJ-2 Section */}
      <div className="px-4 py-3 border-b border-border-subtle">
        <div className="flex items-center gap-1.5 mb-2">
          <Wind className="w-3 h-3 text-accent-orange" />
          <span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">OBJ-2 Spread Sim</span>
        </div>

        {spread ? (
          <>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 mb-2">
              {[
                ['Rate',     `${spread.spread_rate_m_per_min} m/min`],
                ['Area',     `${spread.spread_area_km2} km²`],
                ['Horizon',  `${spread.time_horizon_hrs}h`],
                ['Contain.', `${(spread.containment_probability * 100).toFixed(0)}%`],
                ['Wind',     `${spread.wind_speed_m_s} m/s`],
                ['Conf.',    `${(spread.confidence * 100).toFixed(0)}%`],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-[10px] font-mono text-text-muted">{k}</span>
                  <span className="text-[10px] font-mono text-text-primary">{v}</span>
                </div>
              ))}
            </div>

            <div className="mb-2">
              <div className="text-[9px] font-mono text-text-muted mb-1">Affected cells ({spread.affected_cells.length})</div>
              {spread.affected_cells.map(id => {
                const ac = allCells.find(c => c.grid_id === id);
                return (
                  <div key={id} className="text-[9px] font-mono text-text-secondary leading-relaxed">
                    {ac ? `· ${ac.name}` : `· ${id.slice(0, 12)}…`}
                  </div>
                );
              })}
            </div>

            <div className="px-2 py-1.5 bg-status-partial/10 border border-status-partial/25 rounded">
              <span className="text-[9px] font-mono text-status-partial">
                MANUAL TRIGGER — Cell2Fire not in Airflow DAG
              </span>
            </div>

            {spread.notes && (
              <div className="mt-2 text-[9px] font-mono text-text-muted leading-relaxed">
                {spread.notes}
              </div>
            )}
          </>
        ) : (
          <div className="text-[10px] font-mono text-text-muted">
            {cell.fire_detected_binary === 0 && cell.active_fire_count === 0
              ? 'No active fire — simulation not triggered'
              : 'Simulation pending manual trigger'}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="px-4 py-3 space-y-2">
        {spread && (
          <button
            onClick={() => onNavigate && onNavigate('reports')}
            className="w-full flex items-center justify-between px-3 py-2 bg-accent-blue/10 border border-accent-blue/30 rounded text-accent-blue hover:bg-accent-blue/20 transition-colors"
          >
            <div className="flex items-center gap-1.5">
              <FileText className="w-3 h-3" />
              <span className="text-[10px] font-mono font-semibold">View Incident Report</span>
            </div>
            <ChevronRight className="w-3 h-3" />
          </button>
        )}
        <button
          onClick={() => onNavigate && onNavigate('obj2')}
          className="w-full flex items-center justify-between px-3 py-2 bg-surface-2 border border-border-subtle rounded text-text-secondary hover:text-text-primary hover:bg-surface-3 transition-colors"
        >
          <div className="flex items-center gap-1.5">
            <Wind className="w-3 h-3" />
            <span className="text-[10px] font-mono">Open Spread Simulator</span>
          </div>
          <ChevronRight className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function FireMap({ onNavigate }) {
  const [activeLayer, setActiveLayer] = useState('risk');
  const [selectedId,  setSelectedId]  = useState(null);

  const allCells = useMemo(() => [
    ...CALIFORNIA_CELLS.map(c => ({ ...c, region: 'california' })),
    ...TEXAS_CELLS.map(c => ({ ...c, region: 'texas' })),
  ], []);

  const criticalCount = allCells.filter(c => getRiskTier(c.fire_risk_score) === 'CRITICAL').length;
  const highCount     = allCells.filter(c => getRiskTier(c.fire_risk_score) === 'HIGH').length;
  const activeFireCells = allCells.filter(c => c.fire_detected_binary === 1).length;

  function handleSelect(id) {
    setSelectedId(prev => prev === id ? null : id);
  }

  return (
    <div className="flex h-full overflow-hidden bg-surface-0">
      {/* ── Left: Map canvas ── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center justify-between px-4 py-2 bg-surface-1 border-b border-border-subtle flex-shrink-0">
          <div className="flex items-center gap-3">
            {/* Layer toggle */}
            <div className="flex items-center gap-0.5 bg-surface-2 border border-border-subtle rounded p-0.5">
              {[
                { id: 'risk',   label: 'OBJ-1 Risk',    icon: Activity },
                { id: 'spread', label: 'OBJ-2 Spread',  icon: Wind },
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveLayer(id)}
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-mono transition-colors
                    ${activeLayer === id
                      ? 'bg-surface-3 text-text-primary font-semibold'
                      : 'text-text-muted hover:text-text-secondary'}`}
                >
                  <Icon className="w-3 h-3" />
                  {label}
                </button>
              ))}
            </div>

            {/* Status pills */}
            <div className="flex items-center gap-1.5">
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/15 text-risk-critical border-risk-critical/35">
                {criticalCount} CRITICAL
              </span>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-high/15 text-risk-high border-risk-high/35">
                {highCount} HIGH
              </span>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border bg-risk-critical/15 text-risk-critical border-risk-critical/35">
                <Flame className="w-2.5 h-2.5 inline mr-0.5" />
                {activeFireCells} ACTIVE FIRE
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {activeLayer === 'spread' && (
              <div className="flex items-center gap-1.5 px-2 py-1 bg-status-partial/10 border border-status-partial/25 rounded">
                <Info className="w-2.5 h-2.5 text-status-partial" />
                <span className="text-[9px] font-mono text-status-partial">Cell2Fire: manual trigger only</span>
              </div>
            )}
            <div className="text-[10px] font-mono text-text-muted">
              H3 · {MAP_META.grid_resolution_km}km · {MAP_META.mode}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 px-4 py-1.5 bg-surface-1 border-b border-border-subtle flex-shrink-0">
          <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider">Legend:</span>
          {Object.entries(TIER_CFG).map(([tier, cfg]) => (
            <div key={tier} className="flex items-center gap-1">
              <div className="w-2.5 h-2.5 rounded-sm" style={{ background: cfg.fill, opacity: 0.8 }} />
              <span className="text-[9px] font-mono text-text-muted">{tier}</span>
            </div>
          ))}
          <div className="flex items-center gap-1 ml-2">
            <div className="w-2.5 h-2.5 rounded-full border border-risk-critical animate-pulse" style={{ background: 'transparent' }} />
            <span className="text-[9px] font-mono text-text-muted">Active fire</span>
          </div>
          {activeLayer === 'spread' && (
            <div className="flex items-center gap-1">
              <div className="w-8 h-1.5 rounded" style={{ background: SPREAD_FILL, border: `1px dashed ${SPREAD_STR}` }} />
              <span className="text-[9px] font-mono text-text-muted">Spread area</span>
            </div>
          )}
        </div>

        {/* Maps */}
        <div className="flex-1 overflow-auto p-4">
          <div className="flex flex-wrap gap-6 justify-center">
            <RegionMap
              cells={CALIFORNIA_CELLS}
              bounds={CA_BOUNDS}
              label="California"
              activeLayer={activeLayer}
              selectedId={selectedId}
              spreadData={OBJ2_SPREAD}
              onSelect={handleSelect}
              w={380}
              h={400}
            />
            <RegionMap
              cells={TEXAS_CELLS}
              bounds={TX_BOUNDS}
              label="Texas"
              activeLayer={activeLayer}
              selectedId={selectedId}
              spreadData={OBJ2_SPREAD}
              onSelect={handleSelect}
              w={380}
              h={380}
            />
          </div>

          {/* Map context caption */}
          <div className="mt-4 px-2 text-[9px] font-mono text-text-muted opacity-60">
            Geographic monitoring zones — California &amp; Texas · H3 res-2 · ~64km cells · Outlines approximate
          </div>

          {/* System meta footer */}
          <div className="mt-2 px-2 flex flex-wrap gap-4 text-[9px] font-mono text-text-muted">
            <span>OBJ-1: {MAP_META.obj1_model}</span>
            <span>Last inference: {MAP_META.last_obj1_inference.replace('T',' ').slice(0,19)} UTC</span>
            <span>OBJ-2: {MAP_META.obj2_model}</span>
            <span>Last sim: {MAP_META.last_obj2_simulation.replace('T',' ').slice(0,19)} UTC</span>
            <span className="text-status-partial">OBJ-2 trigger: {MAP_META.obj2_trigger}</span>
          </div>
        </div>
      </div>

      {/* ── Right: Detail panel ── */}
      <div className="w-72 flex-shrink-0 border-l border-border-subtle bg-surface-1 flex flex-col">
        <div className="px-4 py-2.5 border-b border-border-subtle flex items-center gap-2 flex-shrink-0">
          <MapPin className="w-3.5 h-3.5 text-text-muted" />
          <span className="text-[10px] font-mono font-semibold text-text-secondary uppercase tracking-wider">
            Cell Detail
          </span>
          {selectedId && (
            <button
              onClick={() => setSelectedId(null)}
              className="ml-auto text-[9px] font-mono text-text-muted hover:text-text-primary"
            >
              clear
            </button>
          )}
        </div>
        <div className="flex-1 overflow-hidden">
          <DetailPanel
            cellId={selectedId}
            allCells={allCells}
            activeLayer={activeLayer}
            onNavigate={onNavigate}
          />
        </div>
      </div>
    </div>
  );
}
