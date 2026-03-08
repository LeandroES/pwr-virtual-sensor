/**
 * Dashboard — single-page UI for the PWR Digital Twin.
 *
 * Flow
 * ----
 * 1. User fills the SimulationForm and submits.
 * 2. POST /api/runs/ → receives run_id (202 Accepted).
 * 3. react-query polls GET /api/runs/{id}/status every 2 s via refetchInterval.
 * 4. When status === 'completed', the telemetry query activates automatically.
 * 5. TelemetryChart renders the dual-axis response curve.
 */
import { useState, useCallback } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { SimulationForm } from '../components/SimulationForm'
import { StatusBadge } from '../components/StatusBadge'
import { TelemetryChart } from '../components/TelemetryChart'
import { createRun, getRunStatus, getRunTelemetry } from '../api/runs'
import type { RunCreate, RunStatus } from '../types'

// ── Constants ─────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 2_000
const TERMINAL: ReadonlySet<RunStatus> = new Set(['completed', 'failed'])
const NOMINAL_POWER_GW = 3.0

// ── KPI card ──────────────────────────────────────────────────────────────────

interface KpiProps {
  label: string
  value: string
  sub?: string
  accent?: string
}

function KpiCard({ label, value, sub, accent = 'text-gray-800' }: KpiProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
      <p className="text-xs font-medium text-gray-500 leading-none">{label}</p>
      <p className={`mt-2 text-2xl font-bold tabular-nums ${accent}`}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-gray-400">{sub}</p>}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function Dashboard() {
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const queryClient = useQueryClient()

  // ── POST /runs/ ─────────────────────────────────────────────────────────────
  const submitMutation = useMutation({
    mutationFn: createRun,
    onSuccess: (resp) => {
      setActiveRunId(resp.run_id)
      // Discard stale telemetry from any previous run
      queryClient.removeQueries({ queryKey: ['telemetry'] })
      queryClient.removeQueries({ queryKey: ['status'] })
    },
  })

  // ── GET /runs/{id}/status  (adaptive polling) ────────────────────────────────
  const {
    data: statusData,
    isError: statusError,
  } = useQuery({
    queryKey: ['status', activeRunId],
    queryFn: () => getRunStatus(activeRunId!),
    enabled: activeRunId !== null,
    refetchInterval: (query) => {
      const s = query.state.data?.status as RunStatus | undefined
      return s !== undefined && TERMINAL.has(s) ? false : POLL_INTERVAL_MS
    },
    retry: 3,
  })

  const currentStatus = statusData?.status

  // ── GET /runs/{id}/telemetry (fires once when completed) ────────────────────
  const {
    data: telemetryData,
    isLoading: telemetryLoading,
  } = useQuery({
    queryKey: ['telemetry', activeRunId],
    queryFn: () => getRunTelemetry(activeRunId!),
    enabled: activeRunId !== null && currentStatus === 'completed',
    staleTime: Infinity,  // immutable once the worker writes it
    retry: 2,
  })

  // True while a job is in-flight (mutation pending OR status not terminal)
  const isSimulating =
    submitMutation.isPending ||
    currentStatus === 'pending' ||
    currentStatus === 'running'

  // ── Derived KPI values ───────────────────────────────────────────────────────
  const lastPoint = telemetryData?.data.at(-1)
  const finalPowerGW = lastPoint ? lastPoint.power_w / 1e9 : null
  const finalTFuelC  = lastPoint ? lastPoint.t_fuel_k  - 273.15 : null
  const finalTCoolC  = lastPoint ? lastPoint.t_coolant_k - 273.15 : null
  const powerDeltaPct =
    finalPowerGW !== null
      ? (((finalPowerGW - NOMINAL_POWER_GW) / NOMINAL_POWER_GW) * 100).toFixed(2)
      : null

  const handleSubmit = useCallback(
    (payload: RunCreate) => {
      setActiveRunId(null)
      submitMutation.mutate(payload)
    },
    [submitMutation],
  )

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="bg-gray-950 text-white shadow-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-blue-600 shrink-0">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <circle cx="12" cy="12" r="3" strokeWidth="2" />
              <path strokeLinecap="round" strokeWidth="2"
                d="M12 2v2m0 16v2M2 12h2m16 0h2
                   m-3.34-6.66-1.42 1.42M6.76 17.24l-1.42 1.42
                   m0-12.32 1.42 1.42m10.48 10.48 1.42 1.42" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight leading-none">PWR Digital Twin</h1>
            <p className="text-xs text-gray-400 mt-0.5">
              Cinética Puntual · 6 grupos precursores · Modelo térmico lumped
            </p>
          </div>
          <div className="ml-auto hidden sm:flex items-center gap-2 text-xs text-gray-500">
            <span className="w-2 h-2 rounded-full bg-green-400 inline-block"></span>
            API conectada
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 space-y-6 flex-1">

        {/* ── Simulation Form ──────────────────────────────────────────────── */}
        <section className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
          <div className="flex items-center gap-2 mb-5">
            <div className="w-1 h-5 rounded-full bg-blue-600"></div>
            <h2 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">
              Parámetros de simulación
            </h2>
          </div>
          <SimulationForm onSubmit={handleSubmit} isLoading={isSimulating} />

          {submitMutation.isError && (
            <div className="mt-4 flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              <svg className="h-4 w-4 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>
                <strong>Error al enviar:</strong>{' '}
                {submitMutation.error instanceof Error
                  ? submitMutation.error.message
                  : 'Error de red desconocido'}
              </span>
            </div>
          )}
        </section>

        {/* ── Job Status Card ──────────────────────────────────────────────── */}
        {activeRunId && (
          <section className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-1 h-5 rounded-full bg-violet-500"></div>
                  <h2 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">
                    Estado del trabajo
                  </h2>
                </div>
                <p className="font-mono text-xs text-gray-400 pl-3">{activeRunId}</p>
              </div>
              {currentStatus && <StatusBadge status={currentStatus} />}
              {!currentStatus && !statusError && (
                <span className="text-xs text-gray-400 animate-pulse">Conectando…</span>
              )}
            </div>

            {/* Parameters recap */}
            {statusData && (
              <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  {
                    label: 'Reactividad insertada',
                    value: `${(statusData.external_reactivity * 1e5).toFixed(1)} pcm`,
                  },
                  {
                    label: 'Ventana temporal',
                    value: `${statusData.time_span_start} – ${statusData.time_span_end} s`,
                  },
                  {
                    label: 'Δt salida',
                    value: `${statusData.dt} s`,
                  },
                  {
                    label: 'Puntos telemetría',
                    value: telemetryData
                      ? telemetryData.point_count.toLocaleString()
                      : currentStatus === 'completed' ? '…' : '—',
                  },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                    <dt className="text-xs text-gray-500 font-medium">{label}</dt>
                    <dd className="mt-1 text-sm font-semibold text-gray-800 tabular-nums">{value}</dd>
                  </div>
                ))}
              </dl>
            )}

            {/* Worker error */}
            {statusData?.error_message && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                <strong>Error del worker:</strong> {statusData.error_message}
              </div>
            )}
          </section>
        )}

        {/* ── KPI Cards ────────────────────────────────────────────────────── */}
        {telemetryData && telemetryData.data.length > 0 && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 fade-in">
            <KpiCard
              label="Potencia nominal P₀"
              value={`${NOMINAL_POWER_GW.toFixed(3)} GW`}
              sub="Estado estacionario inicial"
              accent="text-gray-700"
            />
            <KpiCard
              label="Potencia nuevo SS"
              value={finalPowerGW !== null ? `${finalPowerGW.toFixed(3)} GW` : '—'}
              sub={
                powerDeltaPct !== null
                  ? `${Number(powerDeltaPct) >= 0 ? '+' : ''}${powerDeltaPct}% respecto P₀`
                  : ''
              }
              accent="text-blue-700"
            />
            <KpiCard
              label="T. Combustible SS"
              value={finalTFuelC !== null ? `${finalTFuelC.toFixed(1)} °C` : '—'}
              sub={
                finalTFuelC !== null
                  ? `ΔT_f = ${(finalTFuelC - (893 - 273.15)).toFixed(1)} K`
                  : ''
              }
              accent="text-red-700"
            />
            <KpiCard
              label="T. Refrigerante SS"
              value={finalTCoolC !== null ? `${finalTCoolC.toFixed(1)} °C` : '—'}
              sub={
                finalTCoolC !== null
                  ? `ΔT_c = ${(finalTCoolC - (593 - 273.15)).toFixed(1)} K`
                  : ''
              }
              accent="text-green-700"
            />
          </div>
        )}

        {/* ── Main Chart ───────────────────────────────────────────────────── */}
        {telemetryData && telemetryData.data.length > 0 && (
          <section className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 fade-in">
            <div className="flex items-center gap-2 mb-5">
              <div className="w-1 h-5 rounded-full bg-green-500"></div>
              <h2 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">
                Respuesta transitoria
              </h2>
              <span className="ml-auto text-xs text-gray-400">
                {telemetryData.point_count.toLocaleString()} puntos · Solver Radau (stiff)
              </span>
            </div>
            <TelemetryChart data={telemetryData.data} />
          </section>
        )}

        {/* ── Telemetry loading spinner ─────────────────────────────────────── */}
        {telemetryLoading && (
          <div className="flex flex-col items-center justify-center py-16 gap-3">
            <div className="animate-spin rounded-full h-10 w-10 border-2 border-gray-200 border-t-blue-600" />
            <p className="text-sm text-gray-500">Cargando telemetría…</p>
          </div>
        )}

        {/* ── Empty state ───────────────────────────────────────────────────── */}
        {!activeRunId && !submitMutation.isPending && (
          <div className="flex flex-col items-center justify-center py-24 text-center gap-4">
            <div className="w-20 h-20 rounded-full bg-gray-100 flex items-center justify-center">
              <svg className="w-9 h-9 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <p className="text-gray-700 font-medium">Sin simulaciones activas</p>
              <p className="text-gray-400 text-sm mt-1">
                Configura un escalón de reactividad y lanza la simulación para visualizar la respuesta transitoria.
              </p>
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex justify-between text-xs text-gray-400">
          <span>PWR Digital Twin · Cinética Puntual + Modelo Lumped</span>
          <span>Solver: Radau · TimescaleDB · Celery</span>
        </div>
      </footer>
    </div>
  )
}
