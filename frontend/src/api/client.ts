import axios from 'axios'

/**
 * Axios instance pre-configured for the PWR Twin API.
 *
 * In development (Vite dev server) all `/api/*` requests are proxied to
 * `http://localhost:8000` with the `/api` prefix stripped — see vite.config.ts.
 *
 * In production (Docker) Nginx forwards `/api/*` → `http://api:8000/` with
 * the same prefix stripping — see nginx.conf.
 *
 * The frontend therefore uses a single consistent base URL regardless of
 * environment, with no CORS issues (same-origin from the browser's view).
 */
const client = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
  timeout: 15_000,
})

export default client
