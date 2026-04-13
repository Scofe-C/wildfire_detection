import { memo } from 'react';

/**
 * <KV label="risk_level" value="HIGH" mono />
 */
const KV = memo(function KV({ label, value, mono = false, className = '' }) {
  return (
    <div className={`flex items-center justify-between py-0.5 ${className}`}>
      <span className="text-[10px] font-mono text-text-muted">{label}</span>
      <span className={`text-[10px] text-text-primary ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  );
});

export default KV;
