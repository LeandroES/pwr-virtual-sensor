/**
 * VirtualSensorDashboard — HMI-style control room panel for the EnKF sensor.
 *
 * Design principles:
 *   • Dark slate theme (slate-950/900/800) — evokes nuclear control room CRTs.
 *   • Cyan accents for EnKF inference.  Amber for sensor noise.  Red for truth.
 *   • Monospaced numerics for all physical values.
 *   • LED-style status indicators with pulse animation.
 *   • ISO-inspired KPI tiles — RMSE, MAE, coverage 68/95%, mean σ.
 *
 * State machine:
 *   idle → submitting → polling (status) → fetching results → displaying
 */
import { useState, useCallback, useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { VirtualSensorChart } from '../components/VirtualSensorChart'
import { submitSensorJob, getSensorStatus, getSensorResults } from '../api/sensor'
import type { SensorSimulateRequest, SensorMetrics, RunStatus } from '../types'

// ── Constants ─────────────────────────────────────────────────────────────────

const POLL_MS    = 2_000
const TERMINAL   = new Set<RunStatus>(['completed', 'failed'])
const NOM_FUEL_C = 893.0  - 273.15   // 619.85 °C
const NOM_COOL_C = 593.0  - 273.15   // 319.85 °C

// Default form values for a compelling, fast demo
interface FormState {
  external_reactivity_pcm: number
  time_span_end:           number
  dt_ms:                   string
  ensemble_size:           number
  obs_noise_std_K:         number
  inflation:               number
  device:                  'cuda' | 'cpu'
}

const FORM_DEFAULTS: FormState = {
  external_reactivity_pcm: 50,   // +50 pcm
  time_span_end:           60,   // 60 s
  dt_ms:                   '10', // dt = 0.010 s
  ensemble_size:           10_000,
  obs_noise_std_K:         3.0,
  inflation:               1.02,
  device:                  'cuda',
}

// ── Sub-components ────────────────────────────────────────────────────────────

// HMI-style LED indicator
function Led({ status }: { status: RunStatus | 'idle' }) {
  const cfg = {
    idle:      { bg: 'bg-slate-600',   ring: '',                 pulse: false },
    pending:   { bg: 'bg-amber-400',   ring: 'ring-amber-400/30', pulse: true  },
    running:   { bg: 'bg-cyan-400',    ring: 'ring-cyan-400/30',  pulse: true  },
    completed: { bg: 'bg-emerald-400', ring: 'ring-emerald-400/25', pulse: false },
    failed:    { bg: 'bg-red-500',     ring: 'ring-red-500/30',   pulse: false },
  }[status]

  return (
    <span
      className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 ${cfg.bg}
        ${cfg.ring ? `ring-4 ${cfg.ring}` : ''}
        ${cfg.pulse ? 'animate-pulse' : ''}`}
    />
  )
}

// HMI numeric tile (oscilloscope-style)
interface MetricTileProps {
  label:   string
  value:   string
  sub?:    string
  accent?: string
  grade?:  'good' | 'warn' | 'bad' | null
}

function MetricTile({ label, value, sub, accent = 'text-cyan-300', grade }: MetricTileProps) {
  const gradeRing = grade === 'good' ? 'border-emerald-500/40'
                  : grade === 'warn' ? 'border-amber-500/40'
                  : grade === 'bad'  ? 'border-red-500/40'
                  :                    'border-slate-700'
  return (
    <div className={`bg-slate-900 border ${gradeRing} rounded-xl p-4 flex flex-col gap-1`}>
      <p className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">{label}</p>
      <p className={`text-2xl font-bold font-mono tabular-nums leading-tight ${accent}`}>{value}</p>
      {sub && <p className="text-[10px] text-slate-600 font-mono">{sub}</p>}
    </div>
  )
}

// Grade helper: how good is coverage compared to theoretical?
function coverageGrade(pct: number, theoretical: number): 'good' | 'warn' | 'bad' {
  const delta = Math.abs(pct - theoretical)
  return delta < 5 ? 'good' : delta < 12 ? 'warn' : 'bad'
}

// ── Form ─────────────────────────────────────────────────────────────────────

interface FormProps {
  onSubmit: (req: SensorSimulateRequest) => void
  disabled: boolean
}

function SensorForm({ onSubmit, disabled }: FormProps) {
  const [state, setState] = useState<FormState>(FORM_DEFAULTS)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const dt = parseFloat(state.dt_ms) / 1000
    onSubmit({
      external_reactivity: state.external_reactivity_pcm * 1e-5,
      time_span:           [0, state.time_span_end],
      dt,
      ensemble_size:       state.ensemble_size,
      obs_noise_std_K:     state.obs_noise_std_K,
      enkf_inflation_factor: state.inflation,
      device:              state.device,
    })
  }

  function field(
    label: string,
    unit: string,
    children: React.ReactNode,
  ) {
    return (
      <label className="flex flex-col gap-1.5">
        <span className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
          {label} <span className="text-slate-600 normal-case font-mono">[{unit}]</span>
        </span>
        {children}
      </label>
    )
  }

  const inputCls = `w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
    text-sm font-mono text-slate-200 focus:outline-none focus:ring-1
    focus:ring-cyan-500 focus:border-cyan-500 transition disabled:opacity-40`

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">

        {field('Reactividad', 'pcm',
          <input
            type="number"
            className={inputCls}
            value={state.external_reactivity_pcm}
            min={-1000} max={1000} step={1}
            disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, external_reactivity_pcm: +e.target.value }))}
          />
        )}

        {field('Duración', 's',
          <input
            type="number"
            className={inputCls}
            value={state.time_span_end}
            min={5} max={3600} step={5}
            disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, time_span_end: +e.target.value }))}
          />
        )}

        {field('Paso dt', 'ms',
          <select
            className={inputCls}
            value={state.dt_ms}
            disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, dt_ms: e.target.value }))}
          >
            <option value="1">1 ms</option>
            <option value="5">5 ms</option>
            <option value="10">10 ms</option>
          </select>
        )}

        {field('Ensemble N', 'miembros',
          <select
            className={inputCls}
            value={state.ensemble_size}
            disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, ensemble_size: +e.target.value }))}
          >
            <option value={1_000}>1 000</option>
            <option value={5_000}>5 000</option>
            <option value={10_000}>10 000</option>
            <option value={50_000}>50 000</option>
            <option value={100_000}>100 000</option>
          </select>
        )}

        {field('Ruido σ_RTD', 'K',
          <input
            type="number"
            className={inputCls}
            value={state.obs_noise_std_K}
            min={0.1} max={50} step={0.5}
            disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, obs_noise_std_K: +e.target.value }))}
          />
        )}

        {field('Dispositivo', '',
          <select
            className={inputCls}
            value={state.device}
            disabled={disabled}
            onChange={(e) =>
              setState((s) => ({ ...s, device: e.target.value as 'cuda' | 'cpu' }))
            }
          >
            <option value="cuda">CUDA / ROCm</option>
            <option value="cpu">CPU (debug)</option>
          </select>
        )}
      </div>

      {/* Noise vs Tuning info strip */}
      <div className="flex items-center gap-3 bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-xs font-mono text-slate-400">
        <span className="shrink-0 text-amber-400 font-semibold">RUIDO</span>
        <span>σ_RTD = {state.obs_noise_std_K.toFixed(1)} K</span>
        <span className="text-slate-600">→</span>
        <span>R = σ² = {(state.obs_noise_std_K ** 2).toFixed(2)} K²</span>
        <span className="text-slate-600 ml-auto">
          ~{Math.round(state.time_span_end / (parseFloat(state.dt_ms) / 1000)).toLocaleString()} pasos · N = {state.ensemble_size.toLocaleString()} miembros
        </span>
      </div>

      <div className="flex justify-end">
        <button
          type="submit"
          disabled={disabled}
          className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg
            bg-cyan-600 hover:bg-cyan-500 active:bg-cyan-700
            text-white text-sm font-semibold tracking-wide
            transition focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-slate-950
            disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {disabled
            ? <><span className="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />Procesando…</>
            : <><span className="text-base">⚡</span> Iniciar análisis EnKF</>
          }
        </button>
      </div>
    </form>
  )
}

// ── KPI strip ─────────────────────────────────────────────────────────────────

function MetricsStrip({ m, obsSigma }: { m: SensorMetrics; obsSigma: number }) {
  const rmseVsSigma = m.rmse_K / obsSigma

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">

      <MetricTile
        label="RMSE T_fuel"
        value={`${m.rmse_K.toFixed(3)} K`}
        sub={`${(rmseVsSigma * 100).toFixed(0)}% de σ_RTD = ${obsSigma.toFixed(1)} K`}
        accent={m.rmse_K < 1 ? 'text-emerald-400' : m.rmse_K < 3 ? 'text-amber-400' : 'text-red-400'}
        grade={m.rmse_K < 1 ? 'good' : m.rmse_K < 3 ? 'warn' : 'bad'}
      />

      <MetricTile
        label="MAE T_fuel"
        value={`${m.mae_K.toFixed(3)} K`}
        sub="Error absoluto medio"
        accent="text-sky-300"
        grade={m.mae_K < 0.8 ? 'good' : m.mae_K < 2.5 ? 'warn' : 'bad'}
      />

      <MetricTile
        label="Cobertura 68%"
        value={`${m.coverage_68pct.toFixed(1)} %`}
        sub="Esperado ≈ 68.3% (±1σ)"
        accent={coverageGrade(m.coverage_68pct, 68.3) === 'good' ? 'text-emerald-400' : 'text-amber-400'}
        grade={coverageGrade(m.coverage_68pct, 68.3)}
      />

      <MetricTile
        label="Cobertura 95%"
        value={`${m.coverage_95pct.toFixed(1)} %`}
        sub="Esperado ≈ 95.4% (±2σ)"
        accent={coverageGrade(m.coverage_95pct, 95.4) === 'good' ? 'text-emerald-400' : 'text-amber-400'}
        grade={coverageGrade(m.coverage_95pct, 95.4)}
      />

      <MetricTile
        label="Incert. media σ"
        value={`${m.mean_ensemble_std_K.toFixed(3)} K`}
        sub="Spread posterior promedio"
        accent="text-slate-300"
      />
    </div>
  )
}

// ── Progress bar while running ────────────────────────────────────────────────

function RunningBanner({ jobId }: { jobId: string }) {
  return (
    <div className="bg-slate-900 border border-cyan-800/50 rounded-xl p-4 flex items-center gap-4">
      <div className="relative shrink-0">
        <div className="w-8 h-8 rounded-full border-2 border-slate-700 border-t-cyan-400 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-cyan-300">EnKF en ejecución</p>
        <p className="text-xs text-slate-500 font-mono truncate">{jobId}</p>
      </div>
      <div className="text-right shrink-0">
        <p className="text-xs text-slate-400">ScipySolver → Ensemble → TimescaleDB</p>
        <p className="text-[10px] text-slate-600 font-mono mt-0.5">sondeo cada {POLL_MS / 1000} s</p>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function VirtualSensorDashboard() {
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [lastNoiseSigma, setLastNoiseSigma] = useState(FORM_DEFAULTS.obs_noise_std_K)
  const queryClient = useQueryClient()

  // ── POST /sensor/simulate ─────────────────────────────────────────────────
  const submitMutation = useMutation({
    mutationFn: submitSensorJob,
    onSuccess: (resp) => {
      setActiveJobId(resp.job_id)
      queryClient.removeQueries({ queryKey: ['sensorStatus'] })
      queryClient.removeQueries({ queryKey: ['sensorResults'] })
    },
  })

  // ── GET /sensor/{id}/status  (adaptive poll) ──────────────────────────────
  const { data: statusData } = useQuery({
    queryKey: ['sensorStatus', activeJobId],
    queryFn: () => getSensorStatus(activeJobId!),
    enabled: activeJobId !== null,
    refetchInterval: (q) => {
      const s = q.state.data?.status as RunStatus | undefined
      return s !== undefined && TERMINAL.has(s) ? false : POLL_MS
    },
    retry: 3,
  })

  const currentStatus = statusData?.status as RunStatus | undefined

  // ── GET /sensor/{id}/results (fires once on completion) ──────────────────
  const { data: resultsData, isLoading: resultsLoading } = useQuery({
    queryKey: ['sensorResults', activeJobId],
    queryFn: () => getSensorResults(activeJobId!, 3_000),
    enabled: activeJobId !== null && currentStatus === 'completed',
    staleTime: Infinity,
    retry: 2,
  })

  const isRunning =
    submitMutation.isPending ||
    currentStatus === 'pending' ||
    currentStatus === 'running'

  // Summary metrics (pre-computed useMemo to avoid recalculation on re-renders)
  const metrics = resultsData?.metrics ?? null
  const chartData = useMemo(() => resultsData?.data ?? [], [resultsData])

  const handleSubmit = useCallback(
    (req: SensorSimulateRequest) => {
      setActiveJobId(null)
      setLastNoiseSigma(req.obs_noise_std_K)
      submitMutation.mutate(req)
    },
    [submitMutation],
  )

  const ledStatus: RunStatus | 'idle' = currentStatus ?? 'idle'

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#020617' }}>

      {/* ── HMI Header ────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800 bg-slate-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center gap-3">

          {/* Reactor icon */}
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-cyan-900/60 border border-cyan-700/40 shrink-0">
            <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <circle cx="12" cy="12" r="3" strokeWidth="2" />
              <path strokeLinecap="round" strokeWidth="1.5"
                d="M12 2v3m0 14v3M2 12h3m14 0h3
                   m-4.22-7.78-2.12 2.12M8.34 15.66l-2.12 2.12
                   m0-11.56 2.12 2.12m7.44 7.44 2.12 2.12" />
            </svg>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <Led status={ledStatus} />
              <h1 className="text-sm font-bold tracking-wider text-slate-100 uppercase font-mono">
                Virtual Sensor · EnKF
              </h1>
              <span className="hidden sm:inline text-[10px] text-slate-600 font-mono border border-slate-800 rounded px-1.5 py-0.5">
                Asimilación de Datos
              </span>
            </div>
            <p className="text-[10px] text-slate-600 font-mono mt-0.5 pl-4">
              Ensemble Kalman Filter · RK4 Vectorizado · ROCm/CUDA · TimescaleDB
            </p>
          </div>

          {/* Status badge */}
          <div className="ml-auto flex items-center gap-3">
            {currentStatus && (
              <span className={`text-xs font-mono font-semibold px-2.5 py-1 rounded border
                ${currentStatus === 'completed' ? 'text-emerald-400 border-emerald-700/50 bg-emerald-950/50'
                : currentStatus === 'running'   ? 'text-cyan-400 border-cyan-700/50 bg-cyan-950/50'
                : currentStatus === 'pending'   ? 'text-amber-400 border-amber-700/50 bg-amber-950/50'
                : 'text-red-400 border-red-700/50 bg-red-950/50'
              }`}>
                {currentStatus.toUpperCase()}
              </span>
            )}
            <div className="hidden sm:flex items-center gap-1.5 text-[10px] text-slate-600 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block"></span>
              API
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 space-y-5 flex-1">

        {/* ── Parameter Configuration ───────────────────────────────────── */}
        <section className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-5">
            <div className="w-1 h-4 rounded-full bg-cyan-500" />
            <h2 className="text-[10px] font-bold tracking-widest text-slate-400 uppercase font-mono">
              Parámetros del Filtro EnKF
            </h2>
          </div>
          <SensorForm onSubmit={handleSubmit} disabled={isRunning} />

          {submitMutation.isError && (
            <div className="mt-4 flex items-start gap-2 p-3 bg-red-950/60 border border-red-800/50 rounded-lg text-xs text-red-300 font-mono">
              <svg className="h-4 w-4 mt-0.5 shrink-0 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>
                <strong className="text-red-400">ERROR:</strong>{' '}
                {submitMutation.error instanceof Error
                  ? submitMutation.error.message
                  : 'Error de red desconocido'}
              </span>
            </div>
          )}
        </section>

        {/* ── Running banner ────────────────────────────────────────────── */}
        {activeJobId && isRunning && <RunningBanner jobId={activeJobId} />}

        {/* ── Worker error ──────────────────────────────────────────────── */}
        {statusData?.error_message && (
          <div className="bg-red-950/60 border border-red-800/50 rounded-xl p-4 text-sm text-red-300 font-mono">
            <span className="text-red-500 font-bold">WORKER ERROR: </span>
            {statusData.error_message}
          </div>
        )}

        {/* ── KPI Metrics Strip ─────────────────────────────────────────── */}
        {metrics && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-1 h-4 rounded-full bg-emerald-500" />
              <h2 className="text-[10px] font-bold tracking-widest text-slate-500 uppercase font-mono">
                Métricas de Rendimiento del Filtro
              </h2>
              <span className="ml-auto text-[10px] text-slate-600 font-mono">
                {metrics.total_points.toLocaleString()} puntos totales
              </span>
            </div>
            <MetricsStrip m={metrics} obsSigma={lastNoiseSigma} />

            {/* Calibration indicator */}
            <div className="mt-3 bg-slate-900 border border-slate-800 rounded-lg px-4 py-2.5
              flex flex-wrap items-center gap-x-6 gap-y-1 text-[10px] font-mono text-slate-500">
              <span className="text-slate-400 font-semibold">CALIBRACIÓN:</span>
              <span>
                Cob. 68%: {
                  coverageGrade(metrics.coverage_68pct, 68.3) === 'good'
                    ? <span className="text-emerald-400">✓ CALIBRADO</span>
                    : <span className="text-amber-400">⚠ DESVIACIÓN {Math.abs(metrics.coverage_68pct - 68.3).toFixed(1)}%</span>
                }
              </span>
              <span>
                Cob. 95%: {
                  coverageGrade(metrics.coverage_95pct, 95.4) === 'good'
                    ? <span className="text-emerald-400">✓ CALIBRADO</span>
                    : <span className="text-amber-400">⚠ DESVIACIÓN {Math.abs(metrics.coverage_95pct - 95.4).toFixed(1)}%</span>
                }
              </span>
              <span className="ml-auto text-slate-600">
                σ_RTD = {lastNoiseSigma.toFixed(1)} K · RMSE/σ = {(metrics.rmse_K / lastNoiseSigma * 100).toFixed(0)}%
              </span>
            </div>
          </section>
        )}

        {/* ── Main Chart ────────────────────────────────────────────────── */}
        {chartData.length > 0 && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-1 h-4 rounded-full bg-sky-400" />
              <h2 className="text-[10px] font-bold tracking-widest text-slate-500 uppercase font-mono">
                Respuesta Transitoria — T_fuel Inferida vs Verdad Real
              </h2>
              {resultsData?.truncated && (
                <span className="ml-auto text-[10px] text-amber-500/70 font-mono">
                  ⚠ Vista reducida · {resultsData.point_count.toLocaleString()} / {resultsData.total_point_count.toLocaleString()} puntos
                </span>
              )}
            </div>

            <VirtualSensorChart
              data={chartData}
              nominalFuelC={NOM_FUEL_C}
              nominalCoolC={NOM_COOL_C}
            />

            {/* Legend explanation */}
            <div className="mt-3 bg-slate-900 border border-slate-800 rounded-lg px-4 py-3
              grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2 text-[10px] font-mono text-slate-500">
              <div className="flex items-center gap-2">
                <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#38bdf8" strokeWidth="2.5" /></svg>
                <span><span className="text-cyan-300">T_fuel inferida</span> — Media posterior del EnKF. Estimación de la T real del combustible.</span>
              </div>
              <div className="flex items-center gap-2">
                <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#f87171" strokeWidth="1.5" strokeDasharray="5 3" /></svg>
                <span><span className="text-red-400">T_fuel real</span> — Verdad oculta (ScipySolver Radau). No disponible en producción.</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block w-6 h-3 rounded-sm" style={{ background: 'rgba(56,189,248,0.14)', border: '1px solid #38bdf8' }} />
                <span><span className="text-sky-300">IC ±2σ (95%)</span> — Intervalo de confianza posterior. Calibrado: ≈95% de la verdad debe estar dentro.</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-400" />
                <span><span className="text-amber-400">RTD ruidoso</span> — T_coolant + ruido Gaussiano σ={lastNoiseSigma.toFixed(1)}K. Única observación del filtro.</span>
              </div>
            </div>
          </section>
        )}

        {/* ── Results loading spinner ───────────────────────────────────── */}
        {resultsLoading && (
          <div className="flex flex-col items-center justify-center py-16 gap-4">
            <div className="relative">
              <div className="w-12 h-12 rounded-full border-2 border-slate-800 border-t-cyan-400 animate-spin" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-3 h-3 rounded-full bg-cyan-600 animate-pulse" />
              </div>
            </div>
            <p className="text-xs text-slate-500 font-mono">Descargando resultados EnKF…</p>
          </div>
        )}

        {/* ── Empty state ───────────────────────────────────────────────── */}
        {!activeJobId && !submitMutation.isPending && (
          <div className="flex flex-col items-center justify-center py-20 text-center gap-5">
            {/* Schematic reactor core icon */}
            <div className="relative w-24 h-24">
              <div className="absolute inset-0 rounded-full border-2 border-slate-800" />
              <div className="absolute inset-3 rounded-full border border-slate-700" />
              <div className="absolute inset-6 rounded-full border border-slate-600" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-4 h-4 rounded-full bg-slate-800 border border-slate-600" />
              </div>
              {/* Orbit dots — simulated ensemble members */}
              {[0, 72, 144, 216, 288].map((deg) => (
                <div
                  key={deg}
                  className="absolute w-1.5 h-1.5 rounded-full bg-slate-600"
                  style={{
                    top: '50%', left: '50%',
                    transform: `rotate(${deg}deg) translateX(36px) translate(-50%, -50%)`,
                  }}
                />
              ))}
            </div>
            <div>
              <p className="text-slate-400 font-semibold font-mono text-sm">
                SENSOR VIRTUAL INACTIVO
              </p>
              <p className="text-slate-600 text-xs font-mono mt-2 max-w-md">
                Configura los parámetros del filtro EnKF y lanza el análisis para ver cómo el
                ensemble de N reactores virtuales infiere la temperatura del combustible a partir
                de la señal ruidosa del RTD de refrigerante.
              </p>
            </div>
            <div className="grid grid-cols-3 gap-4 text-[10px] text-slate-600 font-mono max-w-sm">
              {[
                ['⚡', 'RK4 vectorizado', 'Sin bucles sobre N'],
                ['🎯', 'EnKF estocástico', 'P_f via torch.cov'],
                ['📊', 'TimescaleDB', 'Batch execute_values'],
              ].map(([icon, title, sub]) => (
                <div key={title} className="text-center space-y-1">
                  <div className="text-lg">{icon}</div>
                  <div className="text-slate-500">{title}</div>
                  <div>{sub}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5
          flex flex-wrap justify-between gap-2 text-[10px] font-mono text-slate-700">
          <span>PWR Virtual Sensor · Ensemble Kalman Filter · Asimilación de Datos</span>
          <span>PyTorch (ROCm) · ScipySolver Radau · TimescaleDB Hypertable</span>
        </div>
      </footer>
    </div>
  )
}
