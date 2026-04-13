// ─── Map Helpers ──────────────────────────────────────────────────────────────
// Utilities for the Fire Detection Map: color scales, projection, geometry,
// terrain estimation, crown fire derivation, fire-zone cell generation.

// ─── Color interpolation ──────────────────────────────────────────────────────

function hexToRgb(hex) {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

function rgbToHex(r, g, b) {
  return '#' + [r, g, b]
    .map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0'))
    .join('');
}

function lerpColor(c1, c2, t) {
  const [r1, g1, b1] = hexToRgb(c1);
  const [r2, g2, b2] = hexToRgb(c2);
  return rgbToHex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t);
}

export function interpolateScale(stops, value) {
  if (value <= stops[0][0]) return stops[0][1];
  if (value >= stops[stops.length - 1][0]) return stops[stops.length - 1][1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (value <= stops[i + 1][0]) {
      const t = (value - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return lerpColor(stops[i][1], stops[i + 1][1], t);
    }
  }
  return stops[stops.length - 1][1];
}

// ─── Color scales ─────────────────────────────────────────────────────────────

export function riskColor(score) {
  return interpolateScale([
    [0.0, '#22c55e'], [0.10, '#4ade80'], [0.15, '#a3e635'], [0.25, '#eab308'],
    [0.365, '#f59e0b'], [0.50, '#f97316'], [0.65, '#ef4444'], [0.80, '#dc2626'], [1.0, '#991b1b'],
  ], score);
}

export function moistureColor(rh) {
  return interpolateScale([
    [0, '#78350f'], [15, '#92400e'], [30, '#b45309'], [45, '#d97706'],
    [60, '#a3e635'], [75, '#22c55e'], [90, '#059669'], [100, '#047857'],
  ], rh);
}

export function intensityColor(frp) {
  return interpolateScale([
    [0, '#422006'], [10, '#fef3c7'], [50, '#fbbf24'], [100, '#f97316'],
    [200, '#ef4444'], [350, '#b91c1c'], [500, '#7c3aed'],
  ], frp);
}

export function windSpeedOpacity(speed) {
  return Math.min(1, 0.4 + (speed / 30) * 0.6);
}

// ─── Risk thresholds ──────────────────────────────────────────────────────────

export const RISK_THRESHOLDS = { CRITICAL: 0.65, HIGH: 0.365, MEDIUM: 0.15, LOW: 0.0 };

export function getRiskTier(score) {
  if (score >= RISK_THRESHOLDS.CRITICAL) return 'CRITICAL';
  if (score >= RISK_THRESHOLDS.HIGH)     return 'HIGH';
  if (score >= RISK_THRESHOLDS.MEDIUM)   return 'MEDIUM';
  return 'LOW';
}

export const TIER_COLORS = {
  CRITICAL: { fill: '#ef4444', stroke: '#ef4444', glow: 'rgba(239,68,68,0.5)',  text: 'text-risk-critical' },
  HIGH:     { fill: '#f97316', stroke: '#f97316', glow: 'rgba(249,115,22,0.4)',  text: 'text-risk-high'     },
  MEDIUM:   { fill: '#eab308', stroke: '#eab308', glow: 'rgba(234,179,8,0.3)',   text: 'text-risk-medium'   },
  LOW:      { fill: '#22c55e', stroke: '#22c55e', glow: 'rgba(34,197,94,0.25)',  text: 'text-risk-low'      },
};

// ─── Crown fire (Van Wagner 1977) ────────────────────────────────────────────

export function deriveCrownFire(canopyCoverPct, baseHeightM, bulkDensity) {
  if (!canopyCoverPct || canopyCoverPct < 20) return 'none';
  if (baseHeightM > 6) return 'none';
  if (bulkDensity >= 0.10 && baseHeightM <= 3) return 'active';
  if (baseHeightM <= 6) return 'passive';
  return 'none';
}

export const CROWN_CFG = {
  none:    { label: 'None',    color: 'transparent' },
  passive: { label: 'Passive', color: '#f59e0b' },
  active:  { label: 'Active',  color: '#ef4444' },
};

// ─── Projection ───────────────────────────────────────────────────────────────

export const CA_BOUNDS = { minLat: 32.2, maxLat: 42.2, minLon: -124.8, maxLon: -113.8 };
export const TX_BOUNDS = { minLat: 25.4, maxLat: 37.0, minLon: -107.2, maxLon: -92.8  };

export function project(lat, lon, bounds, w, h) {
  const x = ((lon - bounds.minLon) / (bounds.maxLon - bounds.minLon)) * w;
  const y = ((bounds.maxLat - lat) / (bounds.maxLat - bounds.minLat)) * h;
  return { x, y };
}

// ─── Hex geometry ─────────────────────────────────────────────────────────────

export function hexPoints(cx, cy, r) {
  return [0, 1, 2, 3, 4, 5].map(i => {
    const a = (Math.PI / 180) * (60 * i - 30);
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }).join(' ');
}

export function hexPathD(cx, cy, r) {
  const pts = [0, 1, 2, 3, 4, 5].map(i => {
    const a = (Math.PI / 180) * (60 * i - 30);
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  });
  return `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}` +
    pts.slice(1).map(p => `L${p.x.toFixed(1)},${p.y.toFixed(1)}`).join('') + 'Z';
}

// ─── Wind arrow SVG path (larger, more visible) ──────────────────────────────

export function windArrowD(cx, cy, directionDeg, speed, scale = 1) {
  // Arrow points in the direction wind blows TO (opposite of "from")
  const toDeg = (directionDeg + 180) % 360;
  const rad = (toDeg - 90) * (Math.PI / 180);
  const len = Math.min(scale * (speed / 25) * 16, scale * 20);
  const hl = 5 * scale;

  const sx = cx - (len / 2) * Math.cos(rad);
  const sy = cy - (len / 2) * Math.sin(rad);
  const ex = cx + (len / 2) * Math.cos(rad);
  const ey = cy + (len / 2) * Math.sin(rad);

  // Filled arrowhead triangle
  const h1 = rad + Math.PI * 0.78;
  const h2 = rad - Math.PI * 0.78;
  const hx1 = ex + hl * Math.cos(h1);
  const hy1 = ey + hl * Math.sin(h1);
  const hx2 = ex + hl * Math.cos(h2);
  const hy2 = ey + hl * Math.sin(h2);

  return {
    shaft: `M${sx.toFixed(1)},${sy.toFixed(1)}L${ex.toFixed(1)},${ey.toFixed(1)}`,
    head: `M${ex.toFixed(1)},${ey.toFixed(1)}L${hx1.toFixed(1)},${hy1.toFixed(1)}L${hx2.toFixed(1)},${hy2.toFixed(1)}Z`,
  };
}

export function compassLabel(deg) {
  const dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
  return dirs[Math.round(deg / 22.5) % 16];
}

// ─── Spread direction arrow (from source cell to target cell) ─────────────────

export function spreadArrowD(sx, sy, tx, ty) {
  const dx = tx - sx;
  const dy = ty - sy;
  const len = Math.hypot(dx, dy);
  if (len < 1) return null;
  const nx = dx / len;
  const ny = dy / len;
  // Shorten to not overlap cells
  const margin = 14;
  const ax = sx + nx * margin;
  const ay = sy + ny * margin;
  const bx = tx - nx * margin;
  const by = ty - ny * margin;
  // Arrowhead
  const hl = 6;
  const rad = Math.atan2(by - ay, bx - ax);
  const h1 = rad + Math.PI * 0.8;
  const h2 = rad - Math.PI * 0.8;
  return {
    line: `M${ax.toFixed(1)},${ay.toFixed(1)}L${bx.toFixed(1)},${by.toFixed(1)}`,
    head: `M${bx.toFixed(1)},${by.toFixed(1)}L${(bx + hl * Math.cos(h1)).toFixed(1)},${(by + hl * Math.sin(h1)).toFixed(1)}L${(bx + hl * Math.cos(h2)).toFixed(1)},${(by + hl * Math.sin(h2)).toFixed(1)}Z`,
  };
}

// ─── State outlines ───────────────────────────────────────────────────────────

export const CA_OUTLINE = [
  [42.0,-124.2],[41.4,-124.1],[40.4,-124.4],[39.8,-123.8],
  [38.8,-123.9],[38.3,-123.1],[37.9,-122.7],[37.5,-122.4],
  [37.2,-122.2],[36.6,-121.9],[35.7,-121.3],[35.0,-120.9],
  [34.5,-120.5],[34.1,-119.5],[33.8,-118.5],[33.4,-117.9],
  [32.7,-117.2],[32.5,-117.1],[32.5,-114.7],[33.0,-114.6],
  [34.0,-114.6],[35.2,-114.6],[36.0,-114.7],[37.0,-114.0],
  [38.0,-114.1],[39.0,-114.1],[40.0,-114.1],[41.0,-114.0],
  [42.0,-114.1],[42.0,-124.2],
];

export const TX_OUTLINE = [
  [36.5,-103.0],[36.5,-100.0],[34.0,-100.0],[33.8,-96.5],
  [33.4,-94.0],[31.0,-94.0],[30.0,-93.8],[29.5,-93.9],
  [28.7,-95.7],[27.8,-97.4],[26.2,-97.3],[26.5,-99.1],
  [28.3,-100.3],[29.2,-101.2],[29.8,-103.3],[30.0,-104.6],
  [31.0,-105.3],[31.8,-106.5],[32.0,-106.6],[32.0,-103.0],
  [36.5,-103.0],
];

export const CA_CITIES = [
  { name: 'Los Angeles',   lat: 34.05, lon: -118.24 },
  { name: 'San Francisco', lat: 37.77, lon: -122.42 },
  { name: 'Sacramento',    lat: 38.58, lon: -121.49 },
  { name: 'Santa Barbara', lat: 34.42, lon: -119.70 },
];

export const TX_CITIES = [
  { name: 'Austin',   lat: 30.27, lon: -97.74  },
  { name: 'Dallas',   lat: 32.78, lon: -96.80  },
  { name: 'Houston',  lat: 29.76, lon: -95.37  },
  { name: 'El Paso',  lat: 31.78, lon: -106.40 },
];

// ─── Terrain elevation estimation ────────────────────────────────────────────
// Inverse-distance-weighted blend of known geographic features.
// Produces a plausible topographic backdrop for the SVG maps.

const CA_PEAKS = [
  // Sierra Nevada
  { lat: 36.6, lon: -118.3, elev: 3800, r: 1.8 },
  { lat: 37.8, lon: -119.2, elev: 3200, r: 2.0 },
  { lat: 38.8, lon: -120.0, elev: 2800, r: 1.8 },
  { lat: 40.0, lon: -121.2, elev: 2200, r: 1.5 },
  // Cascades
  { lat: 41.4, lon: -122.2, elev: 2800, r: 1.2 },
  // Coast Ranges
  { lat: 38.5, lon: -123.0, elev: 700,  r: 1.0 },
  { lat: 36.5, lon: -121.5, elev: 800,  r: 0.8 },
  // Transverse Ranges
  { lat: 34.2, lon: -117.5, elev: 2600, r: 1.0 },
  { lat: 34.7, lon: -119.8, elev: 1400, r: 1.0 },
  // Valleys (depressions)
  { lat: 37.5, lon: -120.8, elev: 20,   r: 2.5 },
  { lat: 36.0, lon: -119.5, elev: 30,   r: 2.2 },
  { lat: 38.5, lon: -121.5, elev: 10,   r: 1.8 },
  // Coastal flats
  { lat: 34.0, lon: -118.3, elev: 40,   r: 0.8 },
  { lat: 37.8, lon: -122.4, elev: 10,   r: 0.5 },
  // Desert
  { lat: 35.0, lon: -116.0, elev: 700,  r: 2.0 },
  { lat: 33.3, lon: -115.5, elev: -20,  r: 0.8 },
];

const TX_PEAKS = [
  // Trans-Pecos / Guadalupe
  { lat: 31.9, lon: -104.8, elev: 2400, r: 1.2 },
  { lat: 30.5, lon: -104.0, elev: 1800, r: 1.5 },
  { lat: 29.3, lon: -103.3, elev: 1400, r: 1.0 },
  // High Plains
  { lat: 35.0, lon: -102.0, elev: 1200, r: 2.5 },
  { lat: 33.5, lon: -101.5, elev: 1000, r: 2.0 },
  // Edwards Plateau / Hill Country
  { lat: 30.5, lon: -99.5,  elev: 600,  r: 2.0 },
  { lat: 31.5, lon: -100.0, elev: 700,  r: 1.5 },
  // Gulf Coast (low)
  { lat: 29.0, lon: -95.5,  elev: 5,    r: 2.5 },
  { lat: 27.5, lon: -97.5,  elev: 5,    r: 2.0 },
  { lat: 28.5, lon: -96.5,  elev: 8,    r: 1.5 },
  // East Texas (low rolling)
  { lat: 32.0, lon: -95.0,  elev: 120,  r: 2.0 },
  // Rio Grande Valley
  { lat: 26.5, lon: -98.5,  elev: 30,   r: 1.5 },
  // Central plains
  { lat: 32.0, lon: -97.5,  elev: 200,  r: 2.0 },
];

function estimateElevation(lat, lon, peaks) {
  let sumW = 0;
  let sumE = 0;
  for (const p of peaks) {
    const d = Math.hypot(lat - p.lat, lon - p.lon);
    if (d < 0.01) return p.elev;
    const w = Math.pow(Math.max(0, 1 - d / (p.r * 2)), 2);
    if (w > 0) {
      sumW += w;
      sumE += w * p.elev;
    }
  }
  return sumW > 0 ? sumE / sumW : 150;
}

export function terrainColor(elev) {
  return interpolateScale([
    [-50,  '#081810'],
    [0,    '#0e2216'],
    [100,  '#142a1c'],
    [300,  '#1c3222'],
    [600,  '#243824'],
    [1000, '#2c3420'],
    [1500, '#33301c'],
    [2000, '#382c1a'],
    [2800, '#3e3424'],
    [3500, '#48402e'],
    [4200, '#524a3a'],
  ], elev);
}

// Generate terrain grid: array of { x, y, w, h, fill } in SVG coords
export function generateTerrainGrid(bounds, svgW, svgH, pad, res = 28) {
  const peaks = bounds === CA_BOUNDS ? CA_PEAKS : TX_PEAKS;
  const innerW = svgW - pad * 2;
  const innerH = svgH - pad * 2;
  const cellW = innerW / res;
  const cellH = innerH / res;
  const grid = [];
  for (let row = 0; row < res; row++) {
    for (let col = 0; col < res; col++) {
      const fLat = bounds.maxLat - ((row + 0.5) / res) * (bounds.maxLat - bounds.minLat);
      const fLon = bounds.minLon + ((col + 0.5) / res) * (bounds.maxLon - bounds.minLon);
      const elev = estimateElevation(fLat, fLon, peaks);
      grid.push({
        x: pad + col * cellW,
        y: pad + row * cellH,
        w: cellW + 0.5,
        h: cellH + 0.5,
        fill: terrainColor(elev),
      });
    }
  }
  return grid;
}

// ─── Seeded PRNG ──────────────────────────────────────────────────────────────

function seededRand(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

// ─── Fire-zone 22km cell generator ───────────────────────────────────────────
// Only subdivides cells that are fire-related (active fire, HIGH/CRITICAL risk,
// or in spread-affected zone). Non-fire cells stay at 64km.
// Sub-cells get realistic fire gradient along wind direction.

export function generateFireZoomCells(baseCells, spreadData) {
  const affectedIds = new Set();
  Object.values(spreadData).forEach(sim => sim.affected_cells.forEach(id => affectedIds.add(id)));

  const shouldSubdivide = (cell) =>
    cell.fire_detected_binary === 1 ||
    cell.active_fire_count > 0 ||
    cell.fire_risk_score >= 0.365 ||
    affectedIds.has(cell.grid_id);

  const cells = [];
  const sub = 3;
  const step = 0.16;

  baseCells.forEach((cell, ci) => {
    if (!shouldSubdivide(cell)) {
      cells.push(cell);
      return;
    }

    const rng = seededRand(ci * 1000 + 77);
    const windRad = cell.wind_direction_10m !== undefined
      ? ((cell.wind_direction_10m + 180) % 360 - 90) * (Math.PI / 180)
      : 0;

    for (let di = -(sub - 1); di < sub; di++) {
      for (let dj = -(sub - 1); dj < sub; dj++) {
        const lonStep = step / Math.cos((cell.lat * Math.PI) / 180);
        const lat = cell.lat + di * step;
        const lon = cell.lon + dj * lonStep;

        // Fire risk decays with distance from center, boosted downwind
        const dist = Math.hypot(di, dj);
        const angle = Math.atan2(dj, di);
        const downwind = Math.cos(angle - windRad);
        const riskMod = -dist * 0.06 + (downwind > 0 ? downwind * 0.05 : downwind * 0.02);
        const risk = Math.max(0, Math.min(1, cell.fire_risk_score + riskMod + (rng() - 0.5) * 0.04));

        // Fire presence: center cell keeps original, nearby downwind sub-cells can also burn
        const isCenter = di === 0 && dj === 0;
        const canBurn = cell.fire_detected_binary === 1 && dist <= 1.5 && downwind > 0.2 && rng() > 0.3;
        const fireDetected = isCenter ? cell.fire_detected_binary : (canBurn ? 1 : 0);
        const fireCount = isCenter ? cell.active_fire_count : (canBurn ? Math.ceil(rng() * 2) : 0);
        const frp = fireDetected ? +(cell.mean_frp * (0.4 + rng() * 0.8)).toFixed(1) : 0;

        cells.push({
          ...cell,
          grid_id: isCenter ? cell.grid_id : `${cell.grid_id.slice(0, 10)}_${di}_${dj}`,
          lat, lon,
          name: isCenter ? cell.name : `${cell.name} [${di},${dj}]`,
          fire_risk_score: +risk.toFixed(4),
          temperature_2m: +(cell.temperature_2m + (rng() - 0.5) * 2).toFixed(1),
          relative_humidity_2m: +Math.max(0, Math.min(100, cell.relative_humidity_2m + (rng() - 0.5) * 6)).toFixed(1),
          wind_speed_10m: +Math.max(0, cell.wind_speed_10m + (rng() - 0.5) * 3).toFixed(1),
          wind_direction_10m: cell.wind_direction_10m !== undefined
            ? +((cell.wind_direction_10m + (rng() - 0.5) * 20 + 360) % 360).toFixed(0) : undefined,
          mean_frp: frp,
          fire_detected_binary: fireDetected,
          active_fire_count: fireCount,
          _isSubcell: !isCenter,
        });
      }
    }
  });
  return cells;
}
