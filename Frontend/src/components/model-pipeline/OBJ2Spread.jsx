import { useState, useEffect } from 'react';
import { CheckCircle, Wind, Thermometer, Droplets, Mountain, RefreshCw, AlertTriangle } from 'lucide-react';
import { OBJ2_SIMULATIONS } from '../../data/mockModelData';
import { apiUrl } from '../../api';
import Spinner from '../ui/Spinner';

function KV({ label, value, mono = true, color }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-text-muted text-[10px] w-44 flex-shrink-0">{label}</span>
      <span className={`text-[10px] ${mono ? 'font-mono' : ''} ${color ?? 'text-text-secondary'}`}>{value}</span>
    </div>
  );
}

function SpreadCompass({ deg }) {
  const rad = ((deg - 90) * Math.PI) / 180;
  const cx = 40, cy = 40, r = 28;
  const ex = cx + r * Math.cos(rad);
  const ey = cy + r * Math.sin(rad);
  return (
    <svg width={80} height={80} viewBox="0 0 80 80">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#253348" strokeWidth={1.5} />
      {[['N', 40, 8], ['S', 40, 76], ['E', 74, 44], ['W', 6, 44]].map(([l, x, y]) => (
        <text key={l} x={x} y={y} textAnchor="middle" fontSize={7} fill="#4a5978" fontFamily="monospace">{l}</text>
      ))}
      <line x1={cx} y1={cy} x2={ex} y2={ey} stroke="#ef4444" strokeWidth={2} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={2} fill="#ef4444" />
      <text x={cx} y={cy + 15} textAnchor="middle" fontSize={8} fill="#8a9bbf" fontFamily="monospace">{deg}°</text>
    </svg>
  );
}

function CrownBadge({ status }) {
  const cfg = {
    surface:       { label: 'Surface Fire',       color: 'text-accent-orange bg-accent-orange/10 border-accent-orange/30' },
    passive_crown: { label: 'Passive Crown Fire',  color: 'text-risk-high bg-risk-high/10 border-risk-high/30' },
    active_crown:  { label: 'Active Crown Fire',   color: 'text-risk-critical bg-risk-critical/10 border-risk-critical/30' },
  };
  const c = cfg[status] ?? cfg.surface;
  return (
    <span className={`text-[10px] font-mono font-semibold px-2 py-1 rounded border ${c.color}`}>{c.label}</span>
  );
}

function BurnProbBar({ cellId, prob }) {
  const pct = (prob * 100).toFixed(1);
  const color = prob >= 0.5 ? '#ef4444' : prob >= 0.2 ? '#f59e0b' : '#10b981';
  return (
    <div className="flex items-center gap-2 text-[10px]">
      <span className="font-mono text-text-muted w-36 truncate">{cellId}</span>
      <div className="flex-1 h-2 bg-surface-3 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${prob * 100}%`, background: color }} />
      </div>
      <span className="font-mono text-text-secondary w-12 text-right">{pct}%</span>
    </div>
  );
}

export default function OBJ2Spread() {
  const [liveSims, setLiveSims] = useState({});
  const [loading, setLoading] = useState(true);
  const [activeRegion, setActiveRegion] = useState('california');

  async function fetchSims() {
    setLoading(true);
    const sims = {};
    for (const region of ['california', 'texas']) {
      try {
        const res = await fetch(apiUrl(`/api/spread-simulations?region=${region}`));
        if (res.ok) {
          const data = await res.json();
          sims[region] = data.simulation;
        }
      } catch { /* backend offline */ }
    }
    setLiveSims(sims);
    setLoading(false);
  }

  useEffect(() => { fetchSims(); }, []);

  const sim = liveSims[activeRegion];
  const hasLive = sim && !sim.fallback;
  const mockSim = OBJ2_SIMULATIONS[0];

  // Map live OBJ-2 output to display fields
  const inputs = hasLive ? sim.inputs_used || {} : mockSim?.inputs || {};
  const burnProbs = hasLive ? sim.neighbor_burn_probabilities || {} : {};
  const sortedBurns = Object.entries(burnProbs).sort((a, b) => b[1] - a[1]);

  return (
    <div className="p-6 overflow-y-auto h-full space-y-5">

      {/* Header strip */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Objective</div>
            <div className="text-text-primary text-xs font-semibold">OBJ-2: Fire Spread Simulator</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Physics Model</div>
            <div className="text-text-secondary text-xs font-mono">Rothermel (1972) + Monte Carlo (N=100)</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Data Source</div>
            <div className={`text-xs font-mono font-semibold ${hasLive ? 'text-accent-green' : 'text-accent-orange'}`}>
              {hasLive ? 'Live (GCS)' : 'Mock / Offline'}
            </div>
          </div>
        </div>
        <button onClick={fetchSims} className="flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-mono text-text-muted hover:text-text-primary transition-colors">
          <RefreshCw className="w-3 h-3" />Refresh
        </button>
      </div>

      {/* Region selector */}
      <div className="flex gap-1 bg-surface-2 border border-border-subtle rounded-[7px] p-0.5 w-fit">
        {['california', 'texas'].map(r => (
          <button key={r} onClick={() => setActiveRegion(r)}
            className={`px-3 py-1.5 rounded-[5px] text-[11px] font-mono transition-colors
              ${activeRegion === r ? 'bg-surface-1 text-text-primary shadow-card font-semibold' : 'text-text-muted hover:text-text-secondary'}`}>
            {r.charAt(0).toUpperCase() + r.slice(1)}
            {liveSims[r] && !liveSims[r].fallback && <span className="ml-1 w-1.5 h-1.5 rounded-full bg-accent-green inline-block" />}
          </button>
        ))}
      </div>

      {loading && <div className="flex justify-center py-12"><Spinner /><span className="ml-2 text-text-muted text-xs">Loading simulations...</span></div>}

      {!loading && !hasLive && (
        <div className="bg-surface-2 border border-border-subtle rounded-lg p-6 text-center">
          <AlertTriangle className="w-5 h-5 text-accent-orange mx-auto mb-2" />
          <div className="text-text-secondary text-xs">No live simulation data for {activeRegion}.</div>
          <div className="text-text-muted text-[10px] mt-1">Run the pipeline or click "Run Pipeline Report" in OBJ-3 to generate.</div>
        </div>
      )}

      {!loading && hasLive && (
        <>
          {/* Physics model reference */}
          <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
            <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">Physics Model Stack</div>
            <div className="grid grid-cols-3 gap-3">
              {[
                { title: 'Surface Fire Spread', ref: 'Rothermel (1972)', desc: 'Imperial units: BTU, lb, ft, min. Reaction intensity → rate of spread' },
                { title: 'Fuel Parameters', ref: 'Scott & Burgan FBFM40', desc: '40 fuel models — surface area-to-volume, heat content, moisture of extinction' },
                { title: 'Dead Fuel Moisture', ref: 'Nelson / Simard EMC', desc: 'Equilibrium moisture content from temperature + RH' },
                { title: 'Fireline Intensity', ref: 'Byram (1959)', desc: 'I_B = H × w_c × R  (heat × fuel consumed × rate of spread)' },
                { title: 'Crown Fire Initiation', ref: 'Van Wagner (1977)', desc: 'Critical intensity from canopy base height (CBH)' },
                { title: 'Fire Shape', ref: 'Anderson (1983)', desc: 'Elliptical fire shape via L/B ratio from wind speed' },
              ].map(m => (
                <div key={m.title} className="bg-surface-3 border border-border-subtle rounded p-3">
                  <div className="text-text-primary text-[10px] font-semibold mb-0.5">{m.title}</div>
                  <div className="text-accent-blue text-[9px] font-mono mb-1">{m.ref}</div>
                  <div className="text-text-muted text-[9px]">{m.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Latest simulation */}
          <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider">Latest Simulation — {activeRegion}</div>
                <div className="text-text-muted text-[10px] font-mono">
                  ignition: {sim.ignition_cell} · n={sim.n_simulations} · horizon={sim.horizon_hours}h · {sim.run_timestamp || ''}
                </div>
              </div>
              <CrownBadge status={sim.crown_fire_status} />
            </div>

            <div className="grid grid-cols-3 gap-4">

              {/* Inputs */}
              <div className="bg-surface-3 border border-border-subtle rounded p-3">
                <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Simulation Inputs</div>
                <div className="space-y-1">
                  <KV label="ignition_cell" value={sim.ignition_cell} />
                  <KV label="ignition_probability" value={(sim.ignition_probability || 0).toFixed(3)} color="text-risk-high" />
                  <KV label="n_simulations" value={String(sim.n_simulations)} />
                  <KV label="horizon_hours" value={`${sim.horizon_hours} hr`} />
                  <KV label="dominant_factor" value={sim.dominant_factor || 'N/A'} color="text-accent-blue" />
                </div>
                <div className="mt-3 text-text-muted text-[9px] uppercase tracking-wider mb-1">Weather at Ignition</div>
                <div className="grid grid-cols-2 gap-2 mt-1">
                  {[
                    { icon: Thermometer, label: 'Temp',  value: `${inputs.temperature_c ?? 'N/A'}°C` },
                    { icon: Droplets,    label: 'RH',    value: `${inputs.relative_humidity_pct ?? 'N/A'}%` },
                    { icon: Wind,        label: 'Wind',  value: `${inputs.midflame_wind_mph ?? 'N/A'} mph` },
                    { icon: Mountain,    label: 'Slope', value: `${inputs.ignition_cell_slope_deg ?? 'N/A'}°` },
                  ].map(w => {
                    const W = w.icon;
                    return (
                      <div key={w.label} className="bg-surface-2 border border-border-subtle rounded p-2 flex items-center gap-2">
                        <W className="w-3 h-3 text-text-muted" />
                        <div>
                          <div className="text-[9px] text-text-muted">{w.label}</div>
                          <div className="text-xs font-mono font-semibold text-text-primary">{w.value}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Spread output + compass */}
              <div className="bg-surface-3 border border-border-subtle rounded p-3">
                <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Spread Outputs</div>
                <div className="flex items-center gap-3">
                  <SpreadCompass deg={sim.spread_direction_deg || 0} />
                  <div className="space-y-1">
                    <div>
                      <div className="text-[9px] text-text-muted">spread_direction_deg</div>
                      <div className="text-xl font-mono font-bold text-risk-high">{(sim.spread_direction_deg || 0).toFixed(1)}°</div>
                    </div>
                    <div>
                      <div className="text-[9px] text-text-muted">spread_speed_kmh (p90)</div>
                      <div className="text-lg font-mono font-bold text-risk-high">{(sim.spread_speed_kmh || 0).toFixed(3)} km/h</div>
                    </div>
                  </div>
                </div>
                <div className="mt-3 space-y-1">
                  <KV label="byram_intensity_kwm" value={`${(sim.byram_intensity_kwm || 0).toFixed(1)} kW/m`} color="text-risk-high" />
                  <KV label="dead_fuel_moisture" value={`${(sim.dead_fuel_moisture_pct || 0).toFixed(1)}%`} />
                  <KV label="foliar_moisture" value={`${(sim.foliar_moisture_content_pct || 0).toFixed(1)}%`} />
                  <KV label="speed_mean" value={`${(sim.spread_speed_kmh_mean || 0).toFixed(4)} km/h`} />
                  <KV label="speed_p50" value={`${(sim.spread_speed_kmh_p50 || 0).toFixed(4)} km/h`} />
                  <KV label="speed_p95" value={`${(sim.spread_speed_kmh_p95 || 0).toFixed(4)} km/h`} />
                  <KV label="crown_fire_prob" value={`${((sim.crown_fire_probability || 0) * 100).toFixed(1)}%`} />
                  <KV label="dir_uncertainty" value={`±${(sim.direction_uncertainty_deg || 0).toFixed(1)}°`} />
                </div>
              </div>

              {/* Burn probabilities */}
              <div className="bg-surface-3 border border-border-subtle rounded p-3">
                <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">
                  Neighbor Burn Probabilities ({sortedBurns.length} cells)
                </div>
                {sortedBurns.length > 0 ? (
                  <div className="space-y-1.5 max-h-60 overflow-y-auto">
                    {sortedBurns.map(([cellId, prob]) => (
                      <BurnProbBar key={cellId} cellId={cellId} prob={prob} />
                    ))}
                  </div>
                ) : (
                  <div className="text-text-muted text-[10px] py-4 text-center">No burn probabilities available</div>
                )}
                {sim.max_neighbor_burn_probability != null && (
                  <div className="mt-2 pt-2 border-t border-border-subtle">
                    <KV label="max_burn_probability" value={(sim.max_neighbor_burn_probability).toFixed(4)} color="text-risk-critical" />
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
