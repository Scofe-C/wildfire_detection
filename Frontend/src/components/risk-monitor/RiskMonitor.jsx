import { useState, useEffect } from 'react';
import { AlertTriangle, Wind, Thermometer, Droplets, Flame, MapPin } from 'lucide-react';
import { CALIFORNIA_CELLS, TEXAS_CELLS, getRiskTier, RISK_THRESHOLDS } from '../../data/mockGridData';
import { PIPELINE_META, FUSION_STAGE } from '../../data/mockPipelineData';
import { apiUrl, normalizeCell, fmt } from '../../api';
import LOCATION_NAMES from '../../data/locationNames.json';

// Build a fast lookup: grid_id → location name
const LOCATION_MAP = Object.fromEntries(LOCATION_NAMES.map(l => [l.grid_id, l.location]));

const TIER_COLORS = {
  CRITICAL: { bg: 'bg-risk-critical',     border: 'border-risk-critical',    text: 'text-risk-critical',     badge: 'bg-risk-critical/20 border-risk-critical/40 text-risk-critical' },
  HIGH:     { bg: 'bg-risk-high',          border: 'border-risk-high',        text: 'text-risk-high',          badge: 'bg-risk-high/20 border-risk-high/40 text-risk-high' },
  MEDIUM:   { bg: 'bg-risk-medium',        border: 'border-risk-medium',      text: 'text-risk-medium',        badge: 'bg-risk-medium/20 border-risk-medium/40 text-risk-medium' },
  LOW:      { bg: 'bg-risk-low',           border: 'border-risk-low',         text: 'text-risk-low',           badge: 'bg-risk-low/20 border-risk-low/40 text-risk-low' },
};

function RiskBadge({ tier, size = 'sm' }) {
  const c = TIER_COLORS[tier];
  const sz = size === 'xs' ? 'text-[8px] px-1 py-0.5' : 'text-[9px] px-1.5 py-0.5';
  return (
    <span className={`font-mono font-bold rounded border ${sz} ${c.badge}`}>{tier}</span>
  );
}

function HeatBar({ score }) {
  const tier = getRiskTier(score);
  const colors = { CRITICAL: 'bg-risk-critical', HIGH: 'bg-risk-high', MEDIUM: 'bg-risk-medium', LOW: 'bg-risk-low' };
  return (
    <div className="w-full bg-surface-3 rounded-full h-1.5 overflow-hidden">
      <div className={`h-full ${colors[tier]} rounded-full`} style={{ width: `${score * 100}%` }} />
    </div>
  );
}

function CellCard({ cell, region, onClick, selected }) {
  const tier = getRiskTier(cell.fire_risk_score);
  const c = TIER_COLORS[tier];
  return (
    <button
      onClick={() => onClick(cell, region)}
      className={`text-left p-2.5 rounded border transition-all ${
        selected ? `${c.border} bg-surface-3` : 'border-border-subtle bg-surface-2 hover:bg-surface-3 hover:border-border-default'
      }`}
    >
      <div className="flex items-start justify-between mb-1">
        <div className="text-text-primary text-[10px] font-semibold leading-tight">
          {LOCATION_MAP[cell.grid_id] || cell.grid_id?.slice(0, 10)}
        </div>
        <RiskBadge tier={tier} size="xs" />
      </div>
      <div className={`text-xl font-mono font-bold ${c.text} leading-none mb-1.5`}>
        {fmt(cell.fire_risk_score, 3)}
      </div>
      <HeatBar score={cell.fire_risk_score} />
      <div className="flex justify-between mt-1.5 text-[9px] font-mono text-text-muted">
        <span>{fmt(cell.temperature_2m)}°C</span>
        <span>RH {fmt(cell.relative_humidity_2m)}%</span>
        <span>{fmt(cell.wind_speed_10m)} km/h</span>
        {cell.active_fire_count > 0 && (
          <span className="text-risk-high font-semibold">{cell.active_fire_count} fires</span>
        )}
      </div>
    </button>
  );
}

function CellDetail({ cell, region }) {
  if (!cell) return (
    <div className="flex items-center justify-center h-full text-text-muted text-xs">
      Select a grid cell to view details
    </div>
  );

  const tier = getRiskTier(cell.fire_risk_score);
  const c = TIER_COLORS[tier];

  return (
    <div className="p-4 space-y-4 overflow-y-auto">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <MapPin className="w-3.5 h-3.5 text-text-muted" />
          <span className="text-text-primary text-sm font-semibold">
            {LOCATION_MAP[cell.grid_id] || cell.grid_id}
          </span>
          <RiskBadge tier={tier} />
        </div>
        <div className="text-text-muted text-[10px] font-mono">{cell.grid_id}  ·  {region}  ·  H3 res-2  ·  64 km</div>
        <div className="text-text-muted text-[10px] font-mono">{fmt(cell.lat)}°N  {fmt(Math.abs(cell.lon))}°W</div>
      </div>

      {/* Risk score */}
      <div className={`bg-surface-3 border ${c.border} rounded-lg p-3`}>
        <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">fire_risk_score</div>
        <div className={`text-3xl font-mono font-bold ${c.text}`}>{fmt(cell.fire_risk_score, 4)}</div>
        <HeatBar score={cell.fire_risk_score} />
        <div className="mt-1 text-[9px] text-text-muted font-mono">
          CRITICAL≥0.65  HIGH≥0.365  MEDIUM≥0.15  LOW&lt;0.15
        </div>
      </div>

      {/* Weather features */}
      <div className="bg-surface-3 border border-border-subtle rounded-lg p-3">
        <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Weather Features</div>
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: 'temperature_2m', value: `${fmt(cell.temperature_2m)}°C`, icon: Thermometer },
            { label: 'relative_humidity_2m', value: `${fmt(cell.relative_humidity_2m)}%`, icon: Droplets },
            { label: 'wind_speed_10m', value: `${fmt(cell.wind_speed_10m)} km/h`, icon: Wind },
            { label: 'vpd', value: `${fmt(cell.vpd)} kPa`, icon: Thermometer },
            { label: 'fire_weather_index', value: fmt(cell.fire_weather_index), icon: Flame },
          ].map(f => {
            const I = f.icon;
            return (
              <div key={f.label} className="flex items-center gap-1.5">
                <I className="w-3 h-3 text-text-muted flex-shrink-0" />
                <div>
                  <div className="text-[9px] text-text-muted font-mono">{f.label}</div>
                  <div className="text-[10px] text-text-primary font-mono font-semibold">{f.value}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Fuel & terrain */}
      <div className="bg-surface-3 border border-border-subtle rounded-lg p-3">
        <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Fuel & Terrain</div>
        <div className="space-y-1 text-[10px]">
          {[
            ['fuel_model_fbfm40', cell.fuel_model_fbfm40 || '—'],
            ['elevation_m', cell.elevation_m != null ? `${fmt(cell.elevation_m)} m` : '—'],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="font-mono text-text-muted">{k}</span>
              <span className="font-mono text-text-secondary">{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Fire context */}
      <div className={`border rounded-lg p-3 ${cell.active_fire_count > 0 ? 'bg-risk-high/5 border-risk-high/40' : 'bg-surface-3 border-border-subtle'}`}>
        <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Fire Context</div>
        <div className="space-y-1 text-[10px]">
          {[
            ['active_fire_count', cell.active_fire_count],
            ['fire_detected_binary', cell.fire_detected_binary],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="font-mono text-text-muted">{k}</span>
              <span className={`font-mono font-semibold ${v > 0 ? 'text-risk-high' : 'text-text-secondary'}`}>{v}</span>
            </div>
          ))}
        </div>
        {cell.active_fire_count > 0 && (
          <div className="mt-2 flex items-center gap-1.5">
            <AlertTriangle className="w-3 h-3 text-risk-high" />
            <span className="text-[9px] text-risk-high font-mono">FIRMS hotspots confirmed</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function RiskMonitor() {
  const [selectedCell, setSelectedCell] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [regionFilter, setRegionFilter] = useState('all');
  const [tierFilter, setTierFilter] = useState('all');
  const [liveCells, setLiveCells] = useState(null);
  const [loadingCells, setLoadingCells] = useState(true);

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
          setLiveCells([...ca.cells.map(c => normalizeCell({ ...c, region: 'california' })),
                        ...tx.cells.map(c => normalizeCell({ ...c, region: 'texas' }))]);
        }
      } catch { /* fall back to mock */ }
      if (!cancelled) setLoadingCells(false);
    }
    fetchLive();
    return () => { cancelled = true; };
  }, []);

  const allCells = liveCells || (loadingCells ? [] : [
    ...CALIFORNIA_CELLS.map(c => ({ ...c, region: 'california' })),
    ...TEXAS_CELLS.map(c => ({ ...c, region: 'texas' })),
  ]);

  const filteredCells = allCells.filter(c => {
    const tier = getRiskTier(c.fire_risk_score);
    const regionOk = regionFilter === 'all' || c.region === regionFilter;
    const tierOk = tierFilter === 'all' || tier === tierFilter;
    return regionOk && tierOk;
  }).sort((a, b) => b.fire_risk_score - a.fire_risk_score);

  const counts = {
    CRITICAL: allCells.filter(c => getRiskTier(c.fire_risk_score) === 'CRITICAL').length,
    HIGH:     allCells.filter(c => getRiskTier(c.fire_risk_score) === 'HIGH').length,
    MEDIUM:   allCells.filter(c => getRiskTier(c.fire_risk_score) === 'MEDIUM').length,
    LOW:      allCells.filter(c => getRiskTier(c.fire_risk_score) === 'LOW').length,
  };

  const handleCellClick = (cell, region) => {
    if (selectedCell?.grid_id === cell.grid_id) {
      setSelectedCell(null);
      setSelectedRegion(null);
    } else {
      setSelectedCell(cell);
      setSelectedRegion(region);
    }
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left: grid + filters */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top stats + filters */}
        <div className="p-4 border-b border-border-subtle flex-shrink-0">
          <div className="flex items-center justify-between mb-3">
            {/* Tier counts */}
            <div className="flex items-center gap-3">
              {Object.entries(counts).map(([tier, count]) => {
                const c = TIER_COLORS[tier];
                return (
                  <button
                    key={tier}
                    onClick={() => setTierFilter(tierFilter === tier ? 'all' : tier)}
                    className={`flex items-center gap-1.5 px-2 py-1 rounded border transition-colors ${
                      tierFilter === tier ? `${c.border} ${c.badge}` : 'border-border-subtle bg-surface-2 hover:bg-surface-3'
                    }`}
                  >
                    <span className={`text-lg font-mono font-bold leading-none ${c.text}`}>{count}</span>
                    <span className="text-[9px] font-mono text-text-muted">{tier}</span>
                  </button>
                );
              })}
            </div>

            {/* Region filter */}
            <div className="flex items-center gap-1">
              {['all', 'california', 'texas'].map(r => (
                <button
                  key={r}
                  onClick={() => setRegionFilter(r)}
                  className={`text-[10px] font-mono px-2 py-1 rounded transition-colors ${
                    regionFilter === r
                      ? 'bg-accent-blue/20 border border-accent-blue/40 text-accent-blue'
                      : 'bg-surface-2 border border-border-subtle text-text-muted hover:text-text-secondary hover:bg-surface-3'
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          {/* Grid meta */}
          <div className="flex items-center gap-4 text-[10px] text-text-muted font-mono">
            <span>H3 res-2 · 64 km · {allCells.length} cells total</span>
            <span>CA: {CALIFORNIA_CELLS.length}  TX: {TEXAS_CELLS.length}</span>
            <span>Fire cells: {FUSION_STAGE.fire_cells}</span>
            <span>Mode: <span className="text-accent-green font-semibold">{PIPELINE_META.operational_mode}</span></span>
            <span>Snapshot: 2025-01-15 18:00 UTC</span>
          </div>
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Section headers */}
          {(regionFilter === 'all' || regionFilter === 'california') && (
            <div className="mb-3">
              <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider mb-2 flex items-center gap-2">
                <span>California</span>
                <span className="text-border-default">bbox: [-124.48, 32.53, -114.13, 42.01]</span>
              </div>
              <div className="grid grid-cols-4 gap-2">
                {filteredCells
                  .filter(c => c.region === 'california')
                  .map(cell => (
                    <CellCard
                      key={cell.grid_id}
                      cell={cell}
                      region="california"
                      onClick={handleCellClick}
                      selected={selectedCell?.grid_id === cell.grid_id}
                    />
                  ))}
              </div>
            </div>
          )}

          {(regionFilter === 'all' || regionFilter === 'texas') && (
            <div>
              <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider mb-2 flex items-center gap-2">
                <span>Texas</span>
                <span className="text-border-default">bbox: [-106.65, 25.84, -93.51, 36.50]</span>
              </div>
              <div className="grid grid-cols-4 gap-2">
                {filteredCells
                  .filter(c => c.region === 'texas')
                  .map(cell => (
                    <CellCard
                      key={cell.grid_id}
                      cell={cell}
                      region="texas"
                      onClick={handleCellClick}
                      selected={selectedCell?.grid_id === cell.grid_id}
                    />
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right: detail panel */}
      <div className="w-72 border-l border-border-subtle bg-surface-1 flex-shrink-0 overflow-hidden flex flex-col">
        <div className="px-4 py-3 border-b border-border-subtle flex-shrink-0">
          <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider">Cell Detail</div>
          <div className="text-text-muted text-[9px] font-mono">xgboost_ignition · threshold=0.4596</div>
        </div>
        <div className="flex-1 overflow-y-auto">
          <CellDetail cell={selectedCell} region={selectedRegion} />
        </div>

        {/* Legend */}
        <div className="px-4 py-3 border-t border-border-subtle flex-shrink-0">
          <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Risk Thresholds (model_config.yaml)</div>
          <div className="space-y-1">
            {Object.entries(RISK_THRESHOLDS).reverse().map(([tier, val]) => {
              const c = TIER_COLORS[tier];
              return (
                <div key={tier} className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-sm ${c.bg} flex-shrink-0`} />
                  <span className={`text-[9px] font-mono ${c.text}`}>{tier}</span>
                  <span className="text-[9px] font-mono text-text-muted">
                    {tier === 'LOW' ? `< ${RISK_THRESHOLDS.MEDIUM}` : `≥ ${val}`}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
