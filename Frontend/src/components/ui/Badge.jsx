import { memo } from 'react';

const COLOR_MAP = {
  critical: { outlined: 'bg-risk-critical/10 text-risk-critical border-risk-critical/40', subtle: 'bg-risk-critical/5 text-risk-critical' },
  high:     { outlined: 'bg-risk-high/10 text-risk-high border-risk-high/40',             subtle: 'bg-risk-high/5 text-risk-high' },
  medium:   { outlined: 'bg-risk-medium/10 text-risk-medium border-risk-medium/40',       subtle: 'bg-risk-medium/5 text-risk-medium' },
  low:      { outlined: 'bg-risk-low/10 text-risk-low border-risk-low/40',                subtle: 'bg-risk-low/5 text-risk-low' },
  accent:   { outlined: 'bg-accent/10 text-accent border-accent/40',                      subtle: 'bg-accent/5 text-accent' },
  blue:     { outlined: 'bg-accent-blue/10 text-accent-blue border-accent-blue/40',       subtle: 'bg-accent-blue/5 text-accent-blue' },
  green:    { outlined: 'bg-accent-green/10 text-accent-green border-accent-green/40',    subtle: 'bg-accent-green/5 text-accent-green' },
  orange:   { outlined: 'bg-accent-orange/10 text-accent-orange border-accent-orange/40', subtle: 'bg-accent-orange/5 text-accent-orange' },
  muted:    { outlined: 'bg-surface-3 text-text-secondary border-border-subtle',          subtle: 'bg-surface-2 text-text-muted' },
};

/**
 * <Badge color="critical" variant="outlined">2 CRITICAL</Badge>
 * <Badge color="low" variant="subtle">QUIET</Badge>
 */
const Badge = memo(function Badge({ color = 'muted', variant = 'outlined', className = '', children }) {
  const scheme = COLOR_MAP[color] || COLOR_MAP.muted;
  const cls = variant === 'subtle' ? scheme.subtle : scheme.outlined;
  return (
    <span className={`inline-flex items-center gap-1 text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded-full leading-none
      ${variant === 'outlined' ? 'border' : ''} ${cls} ${className}`}>
      {children}
    </span>
  );
});

export default Badge;
