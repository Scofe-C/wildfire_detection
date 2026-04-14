import { memo } from 'react';

/**
 * <Section title="Pipeline History" icon={Clock} action={<button>...</button>}>
 *   {children}
 * </Section>
 */
const Section = memo(function Section({ title, icon: Icon, action, className = '', children }) {
  return (
    <div className={`rounded-[10px] bg-surface-1 border border-border-subtle shadow-card overflow-hidden ${className}`}>
      <div className="flex items-center justify-between px-3 py-2.5 bg-surface-2 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          {Icon && <Icon className="w-3.5 h-3.5 text-text-muted" />}
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">{title}</h3>
        </div>
        {action}
      </div>
      <div className="p-3">{children}</div>
    </div>
  );
});

export default Section;
