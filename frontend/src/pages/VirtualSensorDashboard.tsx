/**
 * VirtualSensorDashboard — Industrial HMI panel for the EnKF virtual sensor.
 *
 * Design principles:
 *   - Dark slate theme (slate-950/900/800) — nuclear control room aesthetic.
 *   - Cyan accents for EnKF inference. Amber for sensor noise. Red for truth.
 *   - Monospaced numerics for all physical values.
 *   - LED-style status indicators with pulse animation.
 *   - ISO-inspired KPI tiles: RMSE, MAE, coverage 68/95%, mean sigma.
 *   - Zero emojis. SVG icons only.
 *
 * State machine:
 *   idle -> submitting -> polling (status) -> fetching results -> displaying
 */
import { useState, useCallback, useMemo, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'
import { VirtualSensorChart } from '../components/VirtualSensorChart'
import { submitSensorJob, getSensorStatus, getSensorResults } from '../api/sensor'
import type { SensorSimulateRequest, SensorMetrics, SensorResultsResponse, RunStatus } from '../types'
import { LANGUAGES, type LangCode } from '../i18n'
import i18n from '../i18n'

// ── Constants ─────────────────────────────────────────────────────────────────

const POLL_MS    = 2_000
const TERMINAL   = new Set<RunStatus>(['completed', 'failed'])
const NOM_FUEL_C = 893.0 - 273.15   // 619.85 °C
const NOM_COOL_C = 593.0 - 273.15   // 319.85 °C

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
  external_reactivity_pcm: 50,
  time_span_end:           60,
  dt_ms:                   '10',
  ensemble_size:           10_000,
  obs_noise_std_K:         3.0,
  inflation:               1.02,
  device:                  'cuda',
}

// ── Inline SVG icons ──────────────────────────────────────────────────────────

function IconBolt({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path d="M13 2L4.5 13.5H11L9 22L19.5 10.5H13L13 2Z" />
    </svg>
  )
}

function IconCheck({ className = 'w-3.5 h-3.5' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
    </svg>
  )
}

function IconWarning({ className = 'w-3.5 h-3.5' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
    </svg>
  )
}

function IconDownload({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  )
}

function IconCpu({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <rect x="4" y="4" width="16" height="16" rx="2" strokeWidth="1.5" />
      <rect x="8" y="8" width="8" height="8" strokeWidth="1.5" />
      <path d="M9 4V2M15 4V2M9 22v-2M15 22v-2M4 9H2M4 15H2M22 9h-2M22 15h-2" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  )
}

function IconGpu({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <rect x="2" y="7" width="20" height="10" rx="2" strokeWidth="1.5" />
      <path d="M6 11h.01M10 11h.01M14 11h.01M18 11h.01M6 15h.01M10 15h.01M14 15h.01M18 15h.01"
        strokeWidth="2" strokeLinecap="round" />
      <path d="M7 7V5M12 7V5M17 7V5" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  )
}

function IconGlobe({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <circle cx="12" cy="12" r="10" strokeWidth="1.5" />
      <path strokeWidth="1.5" strokeLinecap="round"
        d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10A15.3 15.3 0 0112 2z" />
    </svg>
  )
}

// ── Language selector ─────────────────────────────────────────────────────────

function LanguageSelector() {
  const [current, setCurrent] = useState<LangCode>(i18n.language as LangCode)

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const lang = e.target.value as LangCode
    i18n.changeLanguage(lang)
    setCurrent(lang)
  }

  return (
    <div className="flex items-center gap-1.5">
      <IconGlobe className="w-3.5 h-3.5 text-slate-500 shrink-0" />
      <select
        value={current}
        onChange={handleChange}
        className="bg-slate-900 border border-slate-700 rounded-md px-2 py-1
          text-[10px] font-mono text-slate-400 focus:outline-none focus:ring-1
          focus:ring-cyan-500 focus:border-cyan-500 transition cursor-pointer
          hover:border-slate-600 hover:text-slate-300"
        aria-label="Select language"
      >
        {LANGUAGES.map((l) => (
          <option key={l.code} value={l.code}>{l.label}</option>
        ))}
      </select>
    </div>
  )
}

// ── LED status indicator ───────────────────────────────────────────────────────

function Led({ status }: { status: RunStatus | 'idle' }) {
  const cfg = {
    idle:      { bg: 'bg-slate-600',   ring: '',                  pulse: false },
    pending:   { bg: 'bg-amber-400',   ring: 'ring-amber-400/30', pulse: true  },
    running:   { bg: 'bg-cyan-400',    ring: 'ring-cyan-400/30',  pulse: true  },
    completed: { bg: 'bg-emerald-400', ring: 'ring-emerald-400/25', pulse: false },
    failed:    { bg: 'bg-red-500',     ring: 'ring-red-500/30',   pulse: false },
  }[status]

  return (
    <span
      aria-hidden
      className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 ${cfg.bg}
        ${cfg.ring ? `ring-4 ${cfg.ring}` : ''}
        ${cfg.pulse ? 'animate-pulse' : ''}`}
    />
  )
}

// ── KPI tile ──────────────────────────────────────────────────────────────────

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

function coverageGrade(pct: number, theoretical: number): 'good' | 'warn' | 'bad' {
  const delta = Math.abs(pct - theoretical)
  return delta < 5 ? 'good' : delta < 12 ? 'warn' : 'bad'
}

// ── Parameter form ────────────────────────────────────────────────────────────

interface FormProps {
  onSubmit: (req: SensorSimulateRequest) => void
  disabled: boolean
}

function SensorForm({ onSubmit, disabled }: FormProps) {
  const { t } = useTranslation()
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

  function field(label: string, unit: string, children: React.ReactNode) {
    return (
      <label className="flex flex-col gap-1.5">
        <span className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
          {label}{unit && <span className="text-slate-600 normal-case font-mono"> [{unit}]</span>}
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

        {field(t('form.reactivity'), 'pcm',
          <input type="number" className={inputCls}
            value={state.external_reactivity_pcm} min={-1000} max={1000} step={1}
            disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, external_reactivity_pcm: +e.target.value }))}
          />
        )}

        {field(t('form.duration'), 's',
          <input type="number" className={inputCls}
            value={state.time_span_end} min={5} max={3600} step={5}
            disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, time_span_end: +e.target.value }))}
          />
        )}

        {field(t('form.stepDt'), 'ms',
          <select className={inputCls} value={state.dt_ms} disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, dt_ms: e.target.value }))}
          >
            <option value="1">1 ms</option>
            <option value="5">5 ms</option>
            <option value="10">10 ms</option>
          </select>
        )}

        {field(t('form.ensembleN'), t('form.members'),
          <select className={inputCls} value={state.ensemble_size} disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, ensemble_size: +e.target.value }))}
          >
            <option value={1_000}>1 000</option>
            <option value={5_000}>5 000</option>
            <option value={10_000}>10 000</option>
            <option value={50_000}>50 000</option>
            <option value={100_000}>100 000</option>
          </select>
        )}

        {field(t('form.noiseSigma'), 'K',
          <input type="number" className={inputCls}
            value={state.obs_noise_std_K} min={0.1} max={50} step={0.5}
            disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, obs_noise_std_K: +e.target.value }))}
          />
        )}

        {field(t('form.device'), '',
          <select className={inputCls} value={state.device} disabled={disabled}
            onChange={(e) => setState((s) => ({ ...s, device: e.target.value as 'cuda' | 'cpu' }))}
          >
            <option value="cuda">{t('form.deviceCuda')}</option>
            <option value="cpu">{t('form.deviceCpu')}</option>
          </select>
        )}
      </div>

      <div className="flex items-center gap-3 bg-slate-800/60 border border-slate-700 rounded-lg px-4 py-2.5 text-xs font-mono text-slate-400">
        <span className="shrink-0 text-amber-400 font-semibold">{t('form.noiseLabel')}</span>
        <span>σ_RTD = {state.obs_noise_std_K.toFixed(1)} K</span>
        <span className="text-slate-600">→</span>
        <span>R = σ² = {(state.obs_noise_std_K ** 2).toFixed(2)} K²</span>
        <span className="text-slate-600 ml-auto">
          ~{Math.round(state.time_span_end / (parseFloat(state.dt_ms) / 1000)).toLocaleString()} {t('form.steps')}
          · N = {state.ensemble_size.toLocaleString()} {t('form.members')}
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
          {disabled ? (
            <>
              <span className="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
              {t('form.processing')}
            </>
          ) : (
            <>
              <IconBolt />
              {t('form.submit')}
            </>
          )}
        </button>
      </div>
    </form>
  )
}

// ── KPI metrics strip ─────────────────────────────────────────────────────────

const fmt = (v: number | null, decimals: number, fallback = '—') =>
  v != null && isFinite(v) ? v.toFixed(decimals) : fallback

function MetricsStrip({ m, obsSigma }: { m: SensorMetrics; obsSigma: number }) {
  const { t } = useTranslation()
  const rmseVsSigma = m.rmse_K != null && isFinite(m.rmse_K) ? m.rmse_K / obsSigma : null

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      <MetricTile
        label={t('metrics.rmse')}
        value={`${fmt(m.rmse_K, 3)} K`}
        sub={rmseVsSigma != null
          ? t('metrics.ofSigmaRtd', { pct: (rmseVsSigma * 100).toFixed(0), sigma: obsSigma.toFixed(1) })
          : `σ_RTD = ${obsSigma.toFixed(1)} K`}
        accent={m.rmse_K == null ? 'text-slate-400' : m.rmse_K < 1 ? 'text-emerald-400' : m.rmse_K < 3 ? 'text-amber-400' : 'text-red-400'}
        grade={m.rmse_K == null ? null : m.rmse_K < 1 ? 'good' : m.rmse_K < 3 ? 'warn' : 'bad'}
      />
      <MetricTile
        label={t('metrics.mae')}
        value={`${fmt(m.mae_K, 3)} K`}
        sub={t('metrics.meanAbsError')}
        accent="text-sky-300"
        grade={m.mae_K == null ? null : m.mae_K < 0.8 ? 'good' : m.mae_K < 2.5 ? 'warn' : 'bad'}
      />
      <MetricTile
        label={t('metrics.cov68')}
        value={`${fmt(m.coverage_68pct, 1)} %`}
        sub={t('metrics.expected68')}
        accent={m.coverage_68pct != null ? (coverageGrade(m.coverage_68pct, 68.3) === 'good' ? 'text-emerald-400' : 'text-amber-400') : 'text-slate-400'}
        grade={m.coverage_68pct != null ? coverageGrade(m.coverage_68pct, 68.3) : null}
      />
      <MetricTile
        label={t('metrics.cov95')}
        value={`${fmt(m.coverage_95pct, 1)} %`}
        sub={t('metrics.expected95')}
        accent={m.coverage_95pct != null ? (coverageGrade(m.coverage_95pct, 95.4) === 'good' ? 'text-emerald-400' : 'text-amber-400') : 'text-slate-400'}
        grade={m.coverage_95pct != null ? coverageGrade(m.coverage_95pct, 95.4) : null}
      />
      <MetricTile
        label={t('metrics.meanStd')}
        value={`${fmt(m.mean_ensemble_std_K, 3)} K`}
        sub={t('metrics.posteriorSpread')}
        accent="text-slate-300"
      />
    </div>
  )
}

// ── Smart Switch execution panel ──────────────────────────────────────────────

interface ExecutionPanelProps {
  executionTime: number | null
  deviceUsed:   string | null
  deviceReason: string | null
}

function ExecutionPanel({ executionTime, deviceUsed, deviceReason }: ExecutionPanelProps) {
  const { t } = useTranslation()
  if (!deviceUsed) return null

  const isGpu = deviceUsed.toLowerCase() === 'cuda'

  return (
    <section>
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1 h-4 rounded-full bg-violet-500" />
        <h2 className="text-[10px] font-bold tracking-widest text-slate-500 uppercase font-mono">
          {t('sections.computeEngine')}
        </h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">

        <MetricTile
          label={t('execution.execTime')}
          value={executionTime != null ? `${executionTime.toFixed(2)} s` : '—'}
          sub={t('execution.execSub')}
          accent="text-violet-300"
        />

        <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 flex flex-col gap-2">
          <p className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
            {t('execution.selectedEngine')}
          </p>
          <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-md w-fit
            font-mono font-bold text-sm
            ${isGpu
              ? 'bg-emerald-900/50 border border-emerald-700/50 text-emerald-300'
              : 'bg-sky-900/50 border border-sky-700/50 text-sky-300'
            }`}
          >
            {isGpu ? <IconGpu /> : <IconCpu />}
            {isGpu ? t('execution.gpu') : t('execution.cpu')}
          </div>
          <p className="text-[10px] text-slate-600 font-mono mt-1">
            {isGpu ? t('execution.gpuDesc') : t('execution.cpuDesc')}
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 flex flex-col gap-2">
          <p className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
            {t('execution.selectionReason')}
          </p>
          <p className="text-xs text-slate-300 font-mono leading-relaxed">
            {deviceReason ?? '—'}
          </p>
        </div>
      </div>
    </section>
  )
}

// ── Running banner ────────────────────────────────────────────────────────────

function RunningBanner({ jobId }: { jobId: string }) {
  const { t } = useTranslation()
  return (
    <div className="bg-slate-900 border border-cyan-800/50 rounded-xl p-4 flex items-center gap-4">
      <div className="relative shrink-0">
        <div className="w-8 h-8 rounded-full border-2 border-slate-700 border-t-cyan-400 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-cyan-300">{t('running.title')}</p>
        <p className="text-xs text-slate-500 font-mono truncate">{jobId}</p>
      </div>
      <div className="text-right shrink-0">
        <p className="text-xs text-slate-400">{t('running.subtitle')}</p>
        <p className="text-[10px] text-slate-600 font-mono mt-0.5">
          {t('running.polling', { sec: POLL_MS / 1000 })}
        </p>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function VirtualSensorDashboard() {
  const { t } = useTranslation()
  const [activeJobId, setActiveJobId]       = useState<string | null>(null)
  const [lastNoiseSigma, setLastNoiseSigma] = useState(FORM_DEFAULTS.obs_noise_std_K)
  const [isExporting, setIsExporting]       = useState(false)
  const exportRef                           = useRef<HTMLDivElement>(null)
  const queryClient                         = useQueryClient()

  // ── POST /sensor/simulate ───────────────────────────────────────────────
  const submitMutation = useMutation({
    mutationFn: submitSensorJob,
    onSuccess: (resp) => {
      setActiveJobId(resp.job_id)
      queryClient.removeQueries({ queryKey: ['sensorStatus'] })
      queryClient.removeQueries({ queryKey: ['sensorResults'] })
    },
  })

  // ── GET /sensor/{id}/status (adaptive poll) ────────────────────────────
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

  // ── GET /sensor/{id}/results (fires once on completion) ────────────────
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

  const metrics   = resultsData?.metrics ?? null
  const chartData = useMemo(() => resultsData?.data ?? [], [resultsData])

  const executionTime = resultsData?.execution_time ?? statusData?.execution_time ?? null
  const deviceUsed    = resultsData?.device_used    ?? statusData?.device_used    ?? null
  const deviceReason  = resultsData?.device_reason  ?? statusData?.device_reason  ?? null

  const handleSubmit = useCallback(
    (req: SensorSimulateRequest) => {
      setActiveJobId(null)
      setLastNoiseSigma(req.obs_noise_std_K)
      submitMutation.mutate(req)
    },
    [submitMutation],
  )

  // ── PDF export ──────────────────────────────────────────────────────────
  const handleExportPdf = useCallback(async () => {
    if (!exportRef.current || !activeJobId || isExporting) return
    setIsExporting(true)

    try {
      const canvas = await html2canvas(exportRef.current, {
        scale: 2,
        backgroundColor: '#020617',
        useCORS: true,
        logging: false,
      })

      const imgData = canvas.toDataURL('image/png')
      const A4_W = 210
      const A4_H = 297

      const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })

      const imgW = A4_W
      const imgH = (canvas.height / canvas.width) * A4_W

      let pageTop = 0
      let remaining = imgH
      let page = 0

      while (remaining > 0) {
        if (page > 0) pdf.addPage()
        pdf.addImage(imgData, 'PNG', 0, -pageTop, imgW, imgH)
        pageTop  += A4_H
        remaining -= A4_H
        page++
      }

      pdf.save(`Analisis_SensorVirtual_${activeJobId}.pdf`)
    } finally {
      setIsExporting(false)
    }
  }, [activeJobId, isExporting])

  const ledStatus: RunStatus | 'idle' = currentStatus ?? 'idle'
  const hasResults = currentStatus === 'completed' && (metrics !== null || chartData.length > 0)

  const statusLabel: Record<string, string> = {
    pending:   t('status.pending'),
    running:   t('status.running'),
    completed: t('status.completed'),
    failed:    t('status.failed'),
  }

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#020617' }}>

      {/* ── HMI Header ──────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800 bg-slate-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center gap-3">

          {/* Reactor icon */}
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-cyan-900/60 border border-cyan-700/40 shrink-0">
            <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
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
                {t('header.title')}
              </h1>
              <span className="hidden sm:inline text-[10px] text-slate-600 font-mono border border-slate-800 rounded px-1.5 py-0.5">
                {t('header.tagline')}
              </span>
            </div>
            <p className="text-[10px] text-slate-600 font-mono mt-0.5 pl-4">
              {t('header.subtitle')}
            </p>
          </div>

          {/* Right side: status + language selector + PDF export + API dot */}
          <div className="ml-auto flex items-center gap-3">
            {currentStatus && (
              <span className={`text-xs font-mono font-semibold px-2.5 py-1 rounded border
                ${currentStatus === 'completed' ? 'text-emerald-400 border-emerald-700/50 bg-emerald-950/50'
                : currentStatus === 'running'   ? 'text-cyan-400 border-cyan-700/50 bg-cyan-950/50'
                : currentStatus === 'pending'   ? 'text-amber-400 border-amber-700/50 bg-amber-950/50'
                : 'text-red-400 border-red-700/50 bg-red-950/50'
              }`}>
                {statusLabel[currentStatus] ?? currentStatus.toUpperCase()}
              </span>
            )}

            {hasResults && (
              <button
                onClick={handleExportPdf}
                disabled={isExporting}
                className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg
                  border border-slate-700 bg-slate-800 hover:bg-slate-700
                  text-xs font-semibold text-slate-300 tracking-wide font-mono
                  transition focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 focus:ring-offset-slate-950
                  disabled:opacity-50 disabled:cursor-not-allowed"
                title={`Export job ${activeJobId}`}
              >
                {isExporting ? (
                  <>
                    <span className="w-3.5 h-3.5 rounded-full border border-slate-400/30 border-t-slate-300 animate-spin" />
                    {t('header.generating')}
                  </>
                ) : (
                  <>
                    <IconDownload />
                    {t('header.exportPdf')}
                  </>
                )}
              </button>
            )}

            <LanguageSelector />

            <div className="hidden sm:flex items-center gap-1.5 text-[10px] text-slate-600 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" aria-hidden />
              {t('header.api')}
            </div>
          </div>
        </div>
      </header>

      {/* ── Exportable content area ──────────────────────────────────────── */}
      <div ref={exportRef} style={{ background: '#020617' }}>
        <main className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 space-y-5 flex-1">

          {/* ── Parameter configuration ───────────────────────────────── */}
          <section className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-5">
              <div className="w-1 h-4 rounded-full bg-cyan-500" />
              <h2 className="text-[10px] font-bold tracking-widest text-slate-400 uppercase font-mono">
                {t('sections.filterParams')}
              </h2>
            </div>
            <SensorForm onSubmit={handleSubmit} disabled={isRunning} />

            {submitMutation.isError && (
              <div className="mt-4 flex items-start gap-2 p-3 bg-red-950/60 border border-red-800/50 rounded-lg text-xs text-red-300 font-mono">
                <svg className="h-4 w-4 mt-0.5 shrink-0 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>
                  <strong className="text-red-400">{t('error.error')}</strong>{' '}
                  {submitMutation.error instanceof Error
                    ? submitMutation.error.message
                    : t('error.networkError')}
                </span>
              </div>
            )}
          </section>

          {/* ── Running banner ─────────────────────────────────────────── */}
          {activeJobId && isRunning && <RunningBanner jobId={activeJobId} />}

          {/* ── Worker error ───────────────────────────────────────────── */}
          {statusData?.error_message && (
            <div className="bg-red-950/60 border border-red-800/50 rounded-xl p-4 text-sm text-red-300 font-mono">
              <span className="text-red-500 font-bold">{t('error.workerError')} </span>
              {statusData.error_message}
            </div>
          )}

          {/* ── Smart Switch panel ─────────────────────────────────────── */}
          {(deviceUsed || executionTime != null) && (
            <ExecutionPanel
              executionTime={executionTime}
              deviceUsed={deviceUsed}
              deviceReason={deviceReason}
            />
          )}

          {/* ── KPI metrics strip ──────────────────────────────────────── */}
          {metrics && (
            <section>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-1 h-4 rounded-full bg-emerald-500" />
                <h2 className="text-[10px] font-bold tracking-widest text-slate-500 uppercase font-mono">
                  {t('sections.filterMetrics')}
                </h2>
                <span className="ml-auto text-[10px] text-slate-600 font-mono">
                  {t('metrics.totalPoints', { n: metrics.total_points.toLocaleString() })}
                </span>
              </div>
              <MetricsStrip m={metrics} obsSigma={lastNoiseSigma} />

              {/* Calibration indicator */}
              <div className="mt-3 bg-slate-900 border border-slate-800 rounded-lg px-4 py-2.5
                flex flex-wrap items-center gap-x-6 gap-y-1 text-[10px] font-mono text-slate-500">
                <span className="text-slate-400 font-semibold">{t('calibration.title')}</span>

                <span className="flex items-center gap-1">
                  {t('calibration.cov68')}{' '}
                  {metrics.coverage_68pct == null
                    ? <span className="text-slate-500">—</span>
                    : coverageGrade(metrics.coverage_68pct, 68.3) === 'good'
                      ? <span className="inline-flex items-center gap-1 text-emerald-400">
                          <IconCheck className="w-3 h-3" /> {t('calibration.calibrated')}
                        </span>
                      : <span className="inline-flex items-center gap-1 text-amber-400">
                          <IconWarning className="w-3 h-3" />
                          {t('calibration.deviation')} {Math.abs(metrics.coverage_68pct - 68.3).toFixed(1)}%
                        </span>
                  }
                </span>

                <span className="flex items-center gap-1">
                  {t('calibration.cov95')}{' '}
                  {metrics.coverage_95pct == null
                    ? <span className="text-slate-500">—</span>
                    : coverageGrade(metrics.coverage_95pct, 95.4) === 'good'
                      ? <span className="inline-flex items-center gap-1 text-emerald-400">
                          <IconCheck className="w-3 h-3" /> {t('calibration.calibrated')}
                        </span>
                      : <span className="inline-flex items-center gap-1 text-amber-400">
                          <IconWarning className="w-3 h-3" />
                          {t('calibration.deviation')} {Math.abs(metrics.coverage_95pct - 95.4).toFixed(1)}%
                        </span>
                  }
                </span>

                <span className="ml-auto text-slate-600">
                  σ_RTD = {lastNoiseSigma.toFixed(1)} K · RMSE/σ ={' '}
                  {metrics.rmse_K != null && isFinite(metrics.rmse_K)
                    ? `${(metrics.rmse_K / lastNoiseSigma * 100).toFixed(0)}%`
                    : '—'
                  }
                </span>
              </div>
            </section>
          )}

          {/* ── Main chart ─────────────────────────────────────────────── */}
          {chartData.length > 0 && (
            <section>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-1 h-4 rounded-full bg-sky-400" />
                <h2 className="text-[10px] font-bold tracking-widest text-slate-500 uppercase font-mono">
                  {t('sections.transientResponse')}
                </h2>
                {resultsData?.truncated && (
                  <span className="ml-auto inline-flex items-center gap-1 text-[10px] text-amber-500/70 font-mono">
                    <IconWarning className="w-3 h-3" />
                    {t('chart.truncated', {
                      n:     resultsData.point_count.toLocaleString(),
                      total: resultsData.total_point_count.toLocaleString(),
                    })}
                  </span>
                )}
              </div>

              <VirtualSensorChart
                data={chartData}
                nominalFuelC={NOM_FUEL_C}
                nominalCoolC={NOM_COOL_C}
              />

              {/* Legend */}
              <div className="mt-3 bg-slate-900 border border-slate-800 rounded-lg px-4 py-3
                grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2 text-[10px] font-mono text-slate-500">
                <div className="flex items-center gap-2">
                  <svg width="24" height="8" aria-hidden><line x1="0" y1="4" x2="24" y2="4" stroke="#38bdf8" strokeWidth="2.5" /></svg>
                  <span>
                    <span className="text-cyan-300">{t('legendLabels.inferred')}</span>
                    {' — '}{t('legend.inferredDesc')}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <svg width="24" height="8" aria-hidden><line x1="0" y1="4" x2="24" y2="4" stroke="#f87171" strokeWidth="1.5" strokeDasharray="5 3" /></svg>
                  <span>
                    <span className="text-red-400">{t('legendLabels.true')}</span>
                    {' — '}{t('legend.trueDesc')}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="inline-block w-6 h-3 rounded-sm" style={{ background: 'rgba(56,189,248,0.14)', border: '1px solid #38bdf8' }} aria-hidden />
                  <span>
                    <span className="text-sky-300">{t('legendLabels.ci')}</span>
                    {' — '}{t('legend.ciDesc')}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-400" aria-hidden />
                  <span>
                    <span className="text-amber-400">{t('legendLabels.rtd')}</span>
                    {' — '}{t('legend.rtdDesc', { sigma: lastNoiseSigma.toFixed(1) })}
                  </span>
                </div>
              </div>
            </section>
          )}

          {/* ── Results loading spinner ────────────────────────────────── */}
          {resultsLoading && (
            <div className="flex flex-col items-center justify-center py-16 gap-4">
              <div className="relative">
                <div className="w-12 h-12 rounded-full border-2 border-slate-800 border-t-cyan-400 animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-3 h-3 rounded-full bg-cyan-600 animate-pulse" />
                </div>
              </div>
              <p className="text-xs text-slate-500 font-mono">{t('loading.results')}</p>
            </div>
          )}

          {/* ── Empty state ────────────────────────────────────────────── */}
          {!activeJobId && !submitMutation.isPending && (
            <div className="flex flex-col items-center justify-center py-20 text-center gap-5">
              {/* Schematic reactor core */}
              <div className="relative w-24 h-24" aria-hidden>
                <div className="absolute inset-0 rounded-full border-2 border-slate-800" />
                <div className="absolute inset-3 rounded-full border border-slate-700" />
                <div className="absolute inset-6 rounded-full border border-slate-600" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-4 h-4 rounded-full bg-slate-800 border border-slate-600" />
                </div>
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
                  {t('empty.title')}
                </p>
                <p className="text-slate-600 text-xs font-mono mt-2 max-w-md">
                  {t('empty.desc')}
                </p>
              </div>
              <div className="grid grid-cols-3 gap-6 text-[10px] text-slate-600 font-mono max-w-sm">
                {([
                  {
                    icon: (
                      <svg className="w-6 h-6 mx-auto text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                          d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    ),
                    title: t('empty.rk4Title'),
                    sub:   t('empty.rk4Sub'),
                  },
                  {
                    icon: (
                      <svg className="w-6 h-6 mx-auto text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                        <circle cx="12" cy="12" r="10" strokeWidth="1.5" />
                        <circle cx="12" cy="12" r="6" strokeWidth="1.5" />
                        <circle cx="12" cy="12" r="2" strokeWidth="1.5" />
                        <path d="M12 2v4M12 18v4M2 12h4M18 12h4" strokeWidth="1.5" strokeLinecap="round" />
                      </svg>
                    ),
                    title: t('empty.enkfTitle'),
                    sub:   t('empty.enkfSub'),
                  },
                  {
                    icon: (
                      <svg className="w-6 h-6 mx-auto text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                        <ellipse cx="12" cy="5" rx="9" ry="3" strokeWidth="1.5" />
                        <path strokeWidth="1.5" d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                      </svg>
                    ),
                    title: t('empty.dbTitle'),
                    sub:   t('empty.dbSub'),
                  },
                ] as const).map(({ icon, title, sub }) => (
                  <div key={title} className="text-center space-y-2">
                    {icon}
                    <div className="text-slate-500">{title}</div>
                    <div>{sub}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>
      </div>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5
          flex flex-wrap justify-between gap-2 text-[10px] font-mono text-slate-700">
          <span>{t('footer.left')}</span>
          <span>{t('footer.right')}</span>
        </div>
      </footer>
    </div>
  )
}
