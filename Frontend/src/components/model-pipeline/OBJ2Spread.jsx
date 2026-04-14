import { CheckCircle, Wind, Thermometer, Droplets, Mountain } from 'lucide-react';
import { OBJ2_SIMULATIONS } from '../../data/mockModelData';

function KV({ label, value, mono = true, color }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-text-muted text-[10px] w-44 flex-shrink-0">{label}</span>
      <span className={`text-[10px] ${mono ? 'font-mono' : ''} ${color ?? 'text-text-secondary'}`}>{value}</span>
    </div>
  );
}

function SpreadCompass({ deg }) {
  // Simple SVG compass showing spread direction
  const rad = ((deg - 90) * Math.PI) / 180;
  const cx = 40, cy = 40, r = 28;
  const ex = cx + r * Math.cos(rad);
  const ey = cy + r * Math.sin(rad);
  return (
    <svg width={80} height={80} viewBox="0 0 80 80">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#253348" strokeWidth={1.5} />
      {/* Cardinal labels */}
      {[['N', 40, 8], ['S', 40, 76], ['E', 74, 44], ['W', 6, 44]].map(([l, x, y]) => (
        <text key={l} x={x} y={y} textAnchor="middle" fontSize={7} fill="#4a5978" fontFamily="monospace">{l}</text>
      ))}
      {/* Arrow */}
      <line x1={cx} y1={cy} x2={ex} y2={ey} stroke="#ef4444" strokeWidth={2} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={2} fill="#ef4444" />
      {/* Degree label */}
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

export default function OBJ2Spread() {
  const sim = OBJ2_SIMULATIONS[0];

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
            <div className="text-text-secondary text-xs font-mono">Rothermel (1972) + FBFM40</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">Implementation</div>
            <div className="text-text-secondary text-xs font-mono">PythonFireSpreadSimulator + Cell2Fire C++</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">buffered_iou gate</div>
            <div className="text-accent-green text-xs font-mono font-semibold">≥ 0.35</div>
          </div>
          <div>
            <div className="text-text-muted text-[9px] font-mono uppercase tracking-wider">dice gate</div>
            <div className="text-accent-green text-xs font-mono font-semibold">≥ 0.50</div>
          </div>
        </div>
      </div>

      {/* Physics model reference */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">Physics Model Stack</div>
        <div className="grid grid-cols-3 gap-3">
          {[
            { title: 'Surface Fire Spread', ref: 'Rothermel (1972)', desc: 'Imperial units: BTU, lb, ft, min. Reaction intensity → rate of spread' },
            { title: 'Fuel Parameters', ref: 'Scott & Burgan FBFM40 (RMRS-GTR-153)', desc: '40 Scott & Burgan fuel models — surface area-to-volume, heat content, moisture of extinction' },
            { title: 'Dead Fuel Moisture', ref: 'Nelson / Simard EMC model', desc: 'Piecewise equilibrium moisture content from temperature + RH' },
            { title: 'Fireline Intensity', ref: 'Byram (1959)', desc: 'I_B = H × w_c × R  (heat content × fuel consumed × rate of spread)' },
            { title: 'Crown Fire Initiation', ref: 'Van Wagner (1977)', desc: 'Critical intensity threshold from canopy base height (CBH)' },
            { title: 'Active Crown Fire', ref: 'Scott & Reinhardt (2001)', desc: 'CBD ≥ 0.1 kg/m³ criterion for active crown fire transition' },
            { title: 'Fire Shape', ref: 'Anderson (1983)', desc: 'Elliptical fire shape via L/B ratio from wind speed' },
            { title: 'Wind Reduction', ref: 'Andrews (2012)', desc: 'WAF = 0.4  (10m → midflame wind adjustment factor)' },
            { title: 'Validation Reference', ref: 'CAL FIRE FRAP', desc: 'Historical fire perimeters for buffered IoU / Dice evaluation' },
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
            <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider">Latest Simulation</div>
            <div className="text-text-muted text-[10px] font-mono">{sim.sim_id}  ·  {sim.region}  ·  {sim.timestamp}</div>
          </div>
          <div className="flex items-center gap-2">
            {sim.validation.passed
              ? <><CheckCircle className="w-3.5 h-3.5 text-accent-green" /><span className="text-[10px] font-mono text-accent-green font-semibold">VALIDATION PASS</span></>
              : <><span className="text-[10px] font-mono text-risk-critical font-semibold">VALIDATION FAIL</span></>
            }
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">

          {/* Inputs */}
          <div className="bg-surface-3 border border-border-subtle rounded p-3">
            <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Simulation Inputs</div>
            <div className="space-y-1">
              <KV label="grid_id" value={sim.ignition_grid_id} />
              <KV label="n_simulations" value={String(sim.n_simulations)} />
              <KV label="fire_period_length_hr" value={`${sim.fire_period_length_hr} hr`} />
              <KV label="ignition_probability" value={sim.ignition_probability.toFixed(2)} color="text-risk-high" />
              <KV label="fuel_model_fbfm40" value={sim.inputs.fuel_model_fbfm40} />
              <KV label="elevation_m" value={`${sim.inputs.elevation_m} m`} />
              <KV label="slope_degrees" value={`${sim.inputs.slope_degrees}°`} />
              <KV label="canopy_base_height_m" value={`${sim.inputs.canopy_base_height_m} m`} />
              <KV label="canopy_bulk_density" value={`${sim.inputs.canopy_bulk_density_kgm3} kg/m³`} />
            </div>
          </div>

          {/* Weather at ignition */}
          <div className="bg-surface-3 border border-border-subtle rounded p-3">
            <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Weather at Ignition</div>
            <div className="grid grid-cols-2 gap-2 mt-1">
              {[
                { icon: Thermometer, label: 'Temp', value: `${sim.inputs.temperature_c}°C` },
                { icon: Droplets,    label: 'RH',   value: `${sim.inputs.relative_humidity_pct}%` },
                { icon: Wind,        label: 'Wind', value: `${sim.inputs.wind_speed_kmh} km/h` },
                { icon: Mountain,    label: 'Dir',  value: `${sim.inputs.wind_direction_deg}°` },
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
            <div className="mt-3 text-text-muted text-[9px] uppercase tracking-wider mb-1">Outputs</div>
            <div className="space-y-1">
              <KV label="dead_fuel_moisture_pct" value={`${sim.outputs.dead_fuel_moisture_pct}%`} color="text-risk-high" />
              <KV label="burn_probability_mean" value={sim.outputs.burn_probability_mean.toFixed(2)} color="text-risk-high" />
            </div>
            <div className="mt-2">
              <CrownBadge status={sim.outputs.crown_fire_status} />
            </div>
          </div>

          {/* Spread output + compass */}
          <div className="bg-surface-3 border border-border-subtle rounded p-3">
            <div className="text-text-muted text-[9px] uppercase tracking-wider mb-2">Spread Outputs</div>
            <div className="flex items-center gap-3">
              <SpreadCompass deg={sim.outputs.spread_direction_deg} />
              <div className="space-y-1">
                <div>
                  <div className="text-[9px] text-text-muted">spread_direction_deg</div>
                  <div className="text-xl font-mono font-bold text-risk-high">{sim.outputs.spread_direction_deg}°</div>
                </div>
                <div>
                  <div className="text-[9px] text-text-muted">spread_speed_kmh</div>
                  <div className="text-lg font-mono font-bold text-risk-high">{sim.outputs.spread_speed_kmh} km/h</div>
                </div>
              </div>
            </div>

            <div className="mt-3 pt-2 border-t border-border-subtle">
              <div className="text-text-muted text-[9px] uppercase tracking-wider mb-1">Validation</div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-3 h-3 text-accent-green" />
                  <span className="text-[10px] text-text-secondary">buffered_iou (15% buffer)</span>
                  <span className="text-[10px] font-mono text-accent-green font-semibold">{sim.validation.buffered_iou} ≥ {sim.validation.buffered_iou_threshold}</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-3 h-3 text-accent-green" />
                  <span className="text-[10px] text-text-secondary">dice_coefficient</span>
                  <span className="text-[10px] font-mono text-accent-green font-semibold">{sim.validation.dice_coefficient} ≥ {sim.validation.dice_threshold}</span>
                </div>
                <div className="text-[9px] text-text-muted mt-1">ref: {sim.validation.reference}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration reference */}
      <div className="bg-surface-2 border border-border-subtle rounded-lg p-4">
        <div className="text-text-secondary text-xs font-semibold uppercase tracking-wider mb-3">Configuration (model_config.yaml)</div>
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'n_simulations', values: '50, 100, 200, 500' },
            { label: 'fire_period_length_hr', values: '0.5, 1.0, 2.0, 4.0' },
            { label: 'grid_resolution_m', values: '30, 60, 90' },
            { label: 'weather_duration_hr', values: '24' },
          ].map(c => (
            <div key={c.label} className="bg-surface-3 border border-border-subtle rounded p-2">
              <div className="text-[9px] text-text-muted font-mono">{c.label}</div>
              <div className="text-[10px] text-text-secondary font-mono mt-0.5">[{c.values}]</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
