import { memo } from 'react';

const VARIANT_CLS = {
  default:  'bg-surface-1 border border-border-subtle shadow-card',
  raised:   'bg-surface-1 border border-border-default shadow-card-lg',
  critical: 'bg-surface-1 border border-risk-critical/40 shadow-card glow-critical',
};

/**
 * <Card variant="default|raised|critical">...</Card>
 */
const Card = memo(function Card({ variant = 'default', className = '', children, ...rest }) {
  return (
    <div className={`rounded-[10px] p-3 ${VARIANT_CLS[variant] || VARIANT_CLS.default} ${className}`} {...rest}>
      {children}
    </div>
  );
});

export default Card;
