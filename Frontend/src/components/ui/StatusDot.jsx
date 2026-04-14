import { memo } from 'react';

const STATUS_CFG = {
  working: { color: 'bg-status-working', shadow: '0 0 6px rgba(22,163,74,0.5)',  label: 'WORKING' },
  partial: { color: 'bg-status-partial', shadow: '0 0 6px rgba(202,138,4,0.5)',  label: 'PARTIAL' },
  broken:  { color: 'bg-status-broken',  shadow: '0 0 6px rgba(220,38,38,0.5)',  label: 'BROKEN' },
  planned: { color: 'bg-status-planned', shadow: 'none',                          label: 'PLANNED' },
};

/**
 * <StatusDot status="working" />
 * <StatusDot status="broken" showLabel />
 */
const StatusDot = memo(function StatusDot({ status = 'planned', showLabel = false, className = '' }) {
  const cfg = STATUS_CFG[status] || STATUS_CFG.planned;
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      <span
        className={`w-[7px] h-[7px] rounded-full ${cfg.color} ${status === 'working' ? 'animate-pulse' : ''}`}
        style={{ boxShadow: cfg.shadow }}
      />
      {showLabel && (
        <span className="text-[9px] font-mono font-semibold uppercase tracking-wide text-text-muted">{cfg.label}</span>
      )}
    </span>
  );
});

export default StatusDot;
