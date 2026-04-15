import { useState, useEffect, useRef, useCallback } from 'react';
import { Clock, Bell, AlertTriangle, Sun, Moon, Flame, CheckCircle, X, XCircle, ChevronRight } from 'lucide-react';
import { useTheme } from '../ui/ThemeProvider';
import Badge from '../ui/Badge';
import { apiUrl } from '../../api';

// ── Page titles ──────────────────────────────────────────────────────────────
function getPageTitles(monthYear) {
  return {
    overview:        { label: 'System Overview',              sub: `PyroWatch · wildfire-mlops-123 · us-central1 · ${monthYear}` },
    'data-pipeline': { label: 'Data Pipeline',                sub: 'PyroWatch · wildfire_data_pipeline · 0 */6 * * * UTC' },
    obj1:            { label: 'OBJ-1 Ignition Classifier',    sub: 'PyroWatch · XGBoost + LightGBM · Vertex AI Model Registry' },
    obj2:            { label: 'OBJ-2 Fire Spread Simulator',  sub: 'PyroWatch · Rothermel (1972) + Monte Carlo N=100' },
    obj3:            { label: 'OBJ-3 AI Disaster Reporter',   sub: 'PyroWatch · GeminiDisasterReporter · Vertex AI' },
    'fire-map':      { label: 'Fire Detection Map',           sub: 'PyroWatch · OBJ-1 risk + OBJ-2 spread · H3 64km · CA & TX' },
    'risk-monitor':  { label: 'Risk Monitor',                 sub: `PyroWatch · H3 64km grid · California & Texas · ${monthYear}` },
    reports:         { label: 'Incident Reports',             sub: 'PyroWatch · ICS-209 aligned · OBJ-3 Gemini LLM' },
  };
}

// ── Fallback notifications shown when backend is offline ─────────────────────
const FALLBACK_NOTIFICATIONS = [
  { id: 'fb1', type: 'error',   source: 'System', title: 'Backend offline',          message: 'Cannot reach obj3-dashboard API',        timestamp: '' },
  { id: 'fb2', type: 'warning', source: 'OBJ-1',  title: 'Status unknown',           message: 'Last inference timestamp unavailable',   timestamp: '' },
  { id: 'fb3', type: 'success', source: 'OBJ-3',  title: 'Reports may be available', message: 'Connect backend to load live events',    timestamp: '' },
];

// ── Notification type config ──────────────────────────────────────────────────
const TYPE_CFG = {
  fire:    { icon: Flame,        color: 'text-risk-critical',  bg: 'bg-risk-critical/10',  border: 'border-risk-critical/25' },
  error:   { icon: XCircle,      color: 'text-risk-critical',  bg: 'bg-risk-critical/10',  border: 'border-risk-critical/25' },
  warning: { icon: AlertTriangle, color: 'text-accent-orange', bg: 'bg-accent-orange/10',  border: 'border-accent-orange/25' },
  success: { icon: CheckCircle,  color: 'text-accent-green',   bg: 'bg-accent-green/10',   border: 'border-accent-green/25' },
};

function timeAgo(ts) {
  if (!ts) return '';
  const diff = (Date.now() - new Date(ts).getTime()) / 1000;
  if (diff < 60)    return `${Math.floor(diff)}s ago`;
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

// ── Notification panel ────────────────────────────────────────────────────────
function NotificationPanel({ notifications, readIds, dismissedIds, onMarkAllRead, onDismiss, onClose }) {
  const visible = notifications.filter(n => !dismissedIds.has(n.id));

  return (
    <div
      className="absolute right-0 top-full mt-2 w-[360px] bg-surface-1 border border-border-subtle rounded-xl shadow-2xl z-50 overflow-hidden"
      style={{ boxShadow: '0 8px 40px rgba(0,0,0,0.5)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <span className="text-text-primary text-[13px] font-semibold">Notifications</span>
          {visible.length > 0 && (
            <span className="text-[9px] font-mono bg-surface-3 text-text-muted px-1.5 py-0.5 rounded">
              {visible.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onMarkAllRead}
            className="text-[10px] font-mono text-accent-blue hover:text-accent-blue/80 transition-colors"
          >
            Mark all read
          </button>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* List */}
      <div className="max-h-[420px] overflow-y-auto divide-y divide-border-subtle">
        {visible.length === 0 ? (
          <div className="px-4 py-8 text-center text-text-muted text-[11px] font-mono">
            No notifications
          </div>
        ) : (
          visible.map(n => {
            const cfg = TYPE_CFG[n.type] ?? TYPE_CFG.success;
            const Icon = cfg.icon;
            const unread = !readIds.has(n.id);
            return (
              <div
                key={n.id}
                className={`flex gap-3 px-4 py-3 transition-colors hover:bg-surface-2 ${unread ? 'bg-surface-2/60' : ''}`}
              >
                {/* Icon */}
                <div className={`mt-0.5 flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center ${cfg.bg} border ${cfg.border}`}>
                  <Icon className={`w-3.5 h-3.5 ${cfg.color}`} />
                </div>

                {/* Body */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-[9px] font-mono font-bold text-text-muted uppercase tracking-wider">{n.source}</span>
                      {n.region && (
                        <span className="text-[9px] font-mono text-accent-blue/70 capitalize">{n.region}</span>
                      )}
                      {unread && (
                        <span className="w-1.5 h-1.5 rounded-full bg-accent-blue flex-shrink-0 mt-px" />
                      )}
                    </div>
                    <div className="flex items-center gap-1.5 flex-shrink-0">
                      {n.timestamp && (
                        <span className="text-[9px] font-mono text-text-muted whitespace-nowrap">{timeAgo(n.timestamp)}</span>
                      )}
                      <button
                        onClick={() => onDismiss(n.id)}
                        className="text-text-muted hover:text-text-primary transition-colors opacity-0 group-hover:opacity-100"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                  <div className="text-text-primary text-[12px] font-medium leading-snug mt-0.5">{n.title}</div>
                  <div className="text-text-muted text-[11px] leading-snug mt-0.5 truncate">{n.message}</div>
                </div>

                {/* Dismiss */}
                <button
                  onClick={() => onDismiss(n.id)}
                  className="flex-shrink-0 mt-0.5 text-text-muted hover:text-text-primary transition-colors"
                  title="Dismiss"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* Footer */}
      {visible.length > 0 && (
        <div className="px-4 py-2 border-t border-border-subtle">
          <span className="text-[9px] font-mono text-text-muted">
            Live · polling every 30s · OBJ-1 / OBJ-2 / OBJ-3 events
          </span>
        </div>
      )}
    </div>
  );
}

// ── Header ────────────────────────────────────────────────────────────────────
export default function Header({ activeView }) {
  const { theme, toggle } = useTheme();
  const [now, setNow] = useState(new Date());

  // Notification state
  const [notifications, setNotifications]   = useState([]);
  const [backendOnline, setBackendOnline]    = useState(true);
  const [panelOpen, setPanelOpen]            = useState(false);
  const [readIds, setReadIds]               = useState(() => new Set(JSON.parse(localStorage.getItem('notif_read') || '[]')));
  const [dismissedIds, setDismissedIds]     = useState(() => new Set(JSON.parse(localStorage.getItem('notif_dismissed') || '[]')));

  const bellRef = useRef(null);

  // Live clock
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch notifications
  const fetchNotifications = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/api/notifications'));
      if (!res.ok) throw new Error('bad status');
      const data = await res.json();
      setNotifications(Array.isArray(data) ? data : []);
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
      setNotifications(FALLBACK_NOTIFICATIONS);
    }
  }, []);

  useEffect(() => {
    fetchNotifications();
    const id = setInterval(fetchNotifications, 30_000);
    return () => clearInterval(id);
  }, [fetchNotifications]);

  // Outside click closes panel
  useEffect(() => {
    if (!panelOpen) return;
    function handle(e) {
      if (bellRef.current && !bellRef.current.contains(e.target)) setPanelOpen(false);
    }
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, [panelOpen]);

  // Persist read/dismissed to localStorage
  useEffect(() => {
    localStorage.setItem('notif_read', JSON.stringify([...readIds]));
  }, [readIds]);
  useEffect(() => {
    localStorage.setItem('notif_dismissed', JSON.stringify([...dismissedIds]));
  }, [dismissedIds]);

  // Mark all read when panel opens
  function openPanel() {
    setPanelOpen(v => {
      if (!v) {
        // mark all currently visible as read
        setReadIds(prev => {
          const next = new Set(prev);
          notifications.filter(n => !dismissedIds.has(n.id)).forEach(n => next.add(n.id));
          return next;
        });
      }
      return !v;
    });
  }

  function markAllRead() {
    setReadIds(prev => {
      const next = new Set(prev);
      notifications.forEach(n => next.add(n.id));
      return next;
    });
  }

  function dismiss(id) {
    setDismissedIds(prev => new Set([...prev, id]));
    setReadIds(prev => new Set([...prev, id]));
  }

  const visible    = notifications.filter(n => !dismissedIds.has(n.id));
  const unreadCount = visible.filter(n => !readIds.has(n.id)).length;
  const criticalCount = visible.filter(n => n.type === 'fire' || n.type === 'error').length;
  const warnCount     = visible.filter(n => n.type === 'warning').length;

  const monthYear  = now.toLocaleString('en-US', { month: 'short', year: 'numeric' });
  const PAGE_TITLES = getPageTitles(monthYear);
  const page = PAGE_TITLES[activeView] || PAGE_TITLES.overview;

  return (
    <header className="h-14 bg-surface-1 border-b border-border-subtle flex items-center justify-between px-6 flex-shrink-0">
      <div>
        <h1 className="font-display text-text-primary text-[15px] font-semibold leading-tight">{page.label}</h1>
        <p className="text-text-muted text-[10px] font-mono leading-tight">{page.sub}</p>
      </div>

      <div className="flex items-center gap-3">
        {/* Live alert badges derived from notifications */}
        <div className="flex items-center gap-1.5">
          {criticalCount > 0 && (
            <Badge color="critical">
              <AlertTriangle className="w-2.5 h-2.5" />
              {criticalCount} CRITICAL
            </Badge>
          )}
          {warnCount > 0 && (
            <Badge color="orange">
              <AlertTriangle className="w-2.5 h-2.5" />
              {warnCount} WARN
            </Badge>
          )}
        </div>

        {/* Clock */}
        <div className="flex items-center gap-1.5 text-text-muted">
          <Clock className="w-3 h-3" />
          <span className="text-[10px] font-mono">
            {now.toISOString().replace('T', ' ').slice(0, 19)} UTC
          </span>
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggle}
          className="w-7 h-7 flex items-center justify-center rounded-[7px] border border-border-subtle text-text-muted hover:text-text-primary hover:bg-surface-2 transition-colors"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </button>

        {/* Bell */}
        <div className="relative" ref={bellRef}>
          <button
            onClick={openPanel}
            className={`relative w-7 h-7 flex items-center justify-center rounded-[7px] border transition-colors
              ${panelOpen
                ? 'border-accent-blue/40 bg-accent-blue/10 text-accent-blue'
                : 'border-border-subtle text-text-muted hover:text-text-primary hover:bg-surface-2'
              }`}
            title="Notifications"
          >
            <Bell className="w-3.5 h-3.5" />
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 min-w-[14px] h-[14px] px-0.5 flex items-center justify-center bg-risk-critical text-white text-[8px] font-bold font-mono rounded-full leading-none">
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
            {unreadCount === 0 && !backendOnline && (
              <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-accent-orange rounded-full" />
            )}
          </button>

          {panelOpen && (
            <NotificationPanel
              notifications={notifications}
              readIds={readIds}
              dismissedIds={dismissedIds}
              onMarkAllRead={markAllRead}
              onDismiss={dismiss}
              onClose={() => setPanelOpen(false)}
            />
          )}
        </div>
      </div>
    </header>
  );
}
