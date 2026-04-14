import { useState, useEffect, useCallback, useRef } from 'react';
import { apiUrl } from '../api';

/**
 * Lightweight SWR-style data fetching hook.
 * Prefixes paths with the backend base URL automatically.
 * Stops polling after repeated failures (no console spam when backend is offline).
 *
 *   const { data, error, loading, refresh } = useAPI('/api/status');
 *   const { data } = useAPI('/api/pipeline/history', { interval: 30000 });
 */
export default function useAPI(path, { interval = null, immediate = true } = {}) {
  const [data, setData]       = useState(null);
  const [error, setError]     = useState(null);
  const [loading, setLoading] = useState(immediate);
  const failCount = useRef(0);

  const url = apiUrl(path);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setData(await res.json());
      setError(null);
      failCount.current = 0;
    } catch (e) {
      setError(e);
      failCount.current += 1;
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    if (!immediate) return;
    let cancelled = false;
    let id = null;

    const run = async () => {
      if (cancelled) return;
      await load();
    };
    run();

    if (interval) {
      id = setInterval(() => {
        if (failCount.current >= 2) return; // stop polling when backend is offline
        run();
      }, interval);
    }
    return () => { cancelled = true; if (id) clearInterval(id); };
  }, [load, interval, immediate]);

  const refresh = useCallback(async () => {
    failCount.current = 0;
    await load();
  }, [load]);

  return { data, error, loading, refresh };
}
