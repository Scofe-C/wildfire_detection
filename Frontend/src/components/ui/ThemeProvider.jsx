import { createContext, useContext, useState, useEffect } from 'react';

const ThemeCtx = createContext({ theme: 'dark', toggle: () => {} });

export function useTheme() {
  return useContext(ThemeCtx);
}

export default function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => localStorage.getItem('wf-theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('wf-theme', theme);
  }, [theme]);

  function toggle() {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  }

  return <ThemeCtx.Provider value={{ theme, toggle }}>{children}</ThemeCtx.Provider>;
}
