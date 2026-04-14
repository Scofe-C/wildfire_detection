import { memo } from 'react';

/**
 * <StatCard icon={Flame} label="Critical Cells" value="2" sub="Santa Ynez, Big Bend" critical />
 */
const StatCard = memo(function StatCard({ icon: Icon, label, value, sub, critical = false, className = '' }) {
  return (
    <div className={`rounded-[10px] p-3 border shadow-card transition-all hover:shadow-card-lg hover:-translate-y-0.5
      ${critical
        ? 'bg-surface-1 border-risk-critical/40 glow-critical'
        : 'bg-surface-1 border-border-subtle'
      } ${className}`}
    >
      <div className="flex items-center gap-3">
        {Icon && (
          <div className={`w-9 h-9 rounded-[8px] flex items-center justify-center flex-shrink-0
            ${critical ? 'bg-risk-critical/10' : 'bg-surface-2'}`}>
            <Icon className={`w-4 h-4 ${critical ? 'text-risk-critical' : 'text-text-secondary'}`} />
          </div>
        )}
        <div className="min-w-0">
          <div className={`font-display text-xl font-bold leading-none ${critical ? 'text-risk-critical' : 'text-text-primary'}`}>
            {value}
          </div>
          <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted mt-0.5">{label}</div>
        </div>
      </div>
      {sub && <div className="text-[9px] font-mono text-text-muted mt-2 leading-relaxed truncate">{sub}</div>}
    </div>
  );
});

export default StatCard;
