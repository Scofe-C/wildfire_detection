import { memo } from 'react';

const SIZE_CLS = {
  sm: 'w-3.5 h-3.5 border-[1.5px]',
  md: 'w-5 h-5 border-2',
};

/**
 * <Spinner size="sm" />
 * <Spinner size="md" />
 */
const Spinner = memo(function Spinner({ size = 'md', className = '' }) {
  return (
    <span
      className={`inline-block rounded-full border-text-muted border-t-accent animate-spin ${SIZE_CLS[size] || SIZE_CLS.md} ${className}`}
      role="status"
    />
  );
});

export default Spinner;
