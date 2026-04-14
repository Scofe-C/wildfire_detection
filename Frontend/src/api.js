// Backend API base URL.
// When the FastAPI server is running, requests go directly to it (CORS is enabled).
// When it's offline, fetch simply fails — caught by useAPI hook, no proxy errors.
export const API_BASE = 'http://localhost:8000';

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}
