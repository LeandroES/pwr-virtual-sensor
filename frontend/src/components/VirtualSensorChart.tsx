/**
 * VirtualSensorChart — ComposedChart for the EnKF Virtual Sensor output.
 *
 * Four overlaid data series (bottom → top render order):
 *
 *   1. Area  [ci_upper]   — fills from y-axis floor to +2σ bound  (blue, 18% opacity)
 *   2. Area  [ci_lower]   — masks from y-axis floor to −2σ bound  (bg colour = invisible)
 *        ↳  Together these two areas create the ±2σ confidence band.
 *
 *   3. Line  [tf_true]    — ground-truth T_fuel from ScipySolver   (red dashed)
 *   4. Line  [tf_mean]    — EnKF posterior mean T_fuel              (cyan solid)
 *   5. Line  [noisy_tc]   — noisy RTD coolant readings (dots only)  (amber dots)
 *
 * Dual Y-axis layout:
 *   Left  (yAxisId="fuel")  — T_fuel range  [°C] — inference + CI + ground truth
 *   Right (yAxisId="cool")  — T_coolant range [°C] — RTD readings
 *
 * HMI style: dark slate background, oscilloscope-style grid, monospaced numbers.
 */
import { useMemo } from 'react'
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  type TooltipProps,
} from 'recharts'
import type { SensorResultPoint } from '../types'

// ── Chart background — must match the wrapper div's bg for the CI mask trick ──
const CHART_BG = '#0f172a'   // slate-950

// ── Colour palette ────────────────────────────────────────────────────────────
const C_INFERRED = '#38bdf8'  // sky-400  — EnKF mean T_fuel
const C_TRUE     = '#f87171'  // red-400  — ground truth T_fuel
const C_CI_FILL  = 'rgba(56,189,248,0.14)'   // sky-400 translucent CI band
const C_NOISE    = '#fbbf24'  // amber-400 — noisy RTD dots
const C_GRID     = 'rgba(51,65,85,0.5)'      // slate-700 translucent grid

// ── Derived chart shape ───────────────────────────────────────────────────────

interface ChartRow {
  t:           number   // sim_time_s [s]
  tf_mean:     number   // inferred T_fuel [°C]
  tf_true:     number   // ground-truth T_fuel [°C]
  noisy_tc:    number   // noisy RTD T_coolant [°C]
  ci_upper:    number   // mean + 2σ [°C]
  ci_lower:    number   // mean − 2σ [°C]
}

function toChartRows(data: SensorResultPoint[]): ChartRow[] {
  return data.map((p) => {
    const twoSigma = 2 * p.inferred_t_fuel_std
    return {
      t:        p.sim_time_s,
      tf_mean:  p.inferred_t_fuel_mean - 273.15,
      tf_true:  p.true_t_fuel          - 273.15,
      noisy_tc: p.noisy_t_coolant      - 273.15,
      ci_upper: p.inferred_t_fuel_mean - 273.15 + twoSigma,
      ci_lower: p.inferred_t_fuel_mean - 273.15 - twoSigma,
    }
  })
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

interface TooltipRow {
  name:  string
  value: number
  color: string
  unit:  string
}

function EnKFTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null

  // Build a map from series name to the raw values we actually care about
  const byKey: Record<string, number> = {}
  payload.forEach((e) => { if (e.dataKey) byKey[e.dataKey as string] = e.value as number })

  const rows: TooltipRow[] = [
    { name: 'T_fuel inferida (EnKF)', value: byKey.tf_mean,  color: C_INFERRED, unit: '°C' },
    { name: 'T_fuel real (verdad)',    value: byKey.tf_true,  color: C_TRUE,     unit: '°C' },
    { name: 'RTD ruidoso (T_cool)',    value: byKey.noisy_tc, color: C_NOISE,    unit: '°C' },
    { name: 'IC 95% superior',         value: byKey.ci_upper, color: '#93c5fd',  unit: '°C' },
    { name: 'IC 95% inferior',         value: byKey.ci_lower, color: '#93c5fd',  unit: '°C' },
  ].filter((r) => r.value !== undefined && !isNaN(r.value))

  const error = byKey.tf_mean !== undefined && byKey.tf_true !== undefined
    ? byKey.tf_mean - byKey.tf_true
    : null

  return (
    <div
      className="rounded-lg border border-slate-600 text-xs shadow-xl"
      style={{ background: 'rgba(15,23,42,0.96)', backdropFilter: 'blur(4px)' }}
    >
      <p className="px-3 pt-2.5 pb-1.5 font-mono font-semibold text-cyan-300 border-b border-slate-700">
        t = {Number(label).toFixed(3)} s
      </p>
      <div className="px-3 py-2 space-y-1">
        {rows.map((r) => (
          <div key={r.name} className="flex items-center justify-between gap-5">
            <span style={{ color: r.color }} className="font-medium">{r.name}</span>
            <span className="font-mono text-slate-200 tabular-nums">
              {r.value.toFixed(3)} {r.unit}
            </span>
          </div>
        ))}
        {error !== null && (
          <div className="flex items-center justify-between gap-5 border-t border-slate-700 pt-1 mt-1">
            <span className="text-slate-400">Error instantáneo</span>
            <span
              className="font-mono tabular-nums font-semibold"
              style={{ color: Math.abs(error) < 1 ? '#4ade80' : '#f87171' }}
            >
              {error >= 0 ? '+' : ''}{error.toFixed(3)} K
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Custom legend ─────────────────────────────────────────────────────────────

function EnKFLegend() {
  const items = [
    { label: 'T_fuel inferida (EnKF)',   color: C_INFERRED, lineStyle: 'solid',   shape: 'line' },
    { label: 'IC ±2σ (95%)',             color: C_INFERRED, lineStyle: 'solid',   shape: 'area' },
    { label: 'T_fuel real (ground truth)',color: C_TRUE,     lineStyle: 'dashed',  shape: 'line' },
    { label: 'RTD ruidoso (T_cool)',      color: C_NOISE,    lineStyle: 'scatter', shape: 'dot'  },
  ] as const

  return (
    <div className="flex flex-wrap justify-center gap-x-6 gap-y-2 pt-3 pb-1">
      {items.map(({ label, color, lineStyle, shape }) => (
        <div key={label} className="flex items-center gap-1.5 text-xs text-slate-400">
          {shape === 'area' ? (
            <span
              className="inline-block w-8 h-3.5 rounded-sm"
              style={{ background: C_CI_FILL, border: `1px solid ${color}` }}
            />
          ) : shape === 'dot' ? (
            <span
              className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
              style={{ background: color }}
            />
          ) : (
            <svg width="28" height="10" className="shrink-0">
              <line
                x1="0" y1="5" x2="28" y2="5"
                stroke={color}
                strokeWidth="2"
                strokeDasharray={lineStyle === 'dashed' ? '5 3' : undefined}
              />
            </svg>
          )}
          <span>{label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Domain helpers ────────────────────────────────────────────────────────────

function fuelDomain(rows: ChartRow[]): [number, number] {
  if (rows.length === 0) return [600, 700]
  const lows  = rows.flatMap((r) => [r.ci_lower, r.tf_true])
  const highs = rows.flatMap((r) => [r.ci_upper, r.tf_true])
  const pad = 0.5
  return [Math.floor(Math.min(...lows) - pad), Math.ceil(Math.max(...highs) + pad)]
}

function coolDomain(rows: ChartRow[]): [number, number] {
  if (rows.length === 0) return [310, 340]
  const vals = rows.map((r) => r.noisy_tc)
  const pad = 0.5
  return [Math.floor(Math.min(...vals) - pad), Math.ceil(Math.max(...vals) + pad)]
}

// ── Main export ───────────────────────────────────────────────────────────────

interface Props {
  data: SensorResultPoint[]
  nominalFuelC: number     // nominal T_fuel [°C] for reference line
  nominalCoolC: number     // nominal T_coolant [°C] for reference line
}

export function VirtualSensorChart({ data, nominalFuelC, nominalCoolC }: Props) {
  const rows = useMemo(() => toChartRows(data), [data])
  const fDomain = useMemo(() => fuelDomain(rows), [rows])
  const cDomain = useMemo(() => coolDomain(rows), [rows])

  return (
    // The wrapper div's background MUST match CHART_BG so the CI mask trick works:
    // Area[ci_lower] is filled with CHART_BG, masking the bottom half of
    // Area[ci_upper], creating the visual appearance of a band from −2σ to +2σ.
    <div className="rounded-xl overflow-hidden" style={{ background: CHART_BG }}>
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart
          data={rows}
          margin={{ top: 16, right: 72, left: 10, bottom: 28 }}
        >
          {/* ── Grid ─────────────────────────────────────────────────────── */}
          <CartesianGrid
            strokeDasharray="2 4"
            stroke={C_GRID}
            vertical={true}
          />

          {/* ── Axes ─────────────────────────────────────────────────────── */}
          <XAxis
            dataKey="t"
            type="number"
            domain={['dataMin', 'dataMax']}
            tick={{ fontSize: 10, fill: '#94a3b8', fontFamily: 'monospace' }}
            tickFormatter={(v: number) => `${v.toFixed(1)}s`}
            tickLine={{ stroke: '#475569' }}
            axisLine={{ stroke: '#475569' }}
            label={{
              value: 'Tiempo [s]',
              position: 'insideBottom',
              offset: -14,
              fontSize: 10,
              fill: '#64748b',
              fontFamily: 'monospace',
            }}
          />

          {/* Left Y: T_fuel [°C] */}
          <YAxis
            yAxisId="fuel"
            orientation="left"
            domain={fDomain}
            tick={{ fontSize: 10, fill: '#38bdf8', fontFamily: 'monospace' }}
            tickFormatter={(v: number) => `${v.toFixed(1)}`}
            tickLine={{ stroke: '#475569' }}
            axisLine={{ stroke: '#475569' }}
            width={52}
            label={{
              value: 'T_fuel [°C]',
              angle: -90,
              position: 'insideLeft',
              offset: 6,
              fontSize: 10,
              fill: '#38bdf8',
              fontFamily: 'monospace',
            }}
          />

          {/* Right Y: T_coolant [°C] */}
          <YAxis
            yAxisId="cool"
            orientation="right"
            domain={cDomain}
            tick={{ fontSize: 10, fill: '#fbbf24', fontFamily: 'monospace' }}
            tickFormatter={(v: number) => `${v.toFixed(1)}`}
            tickLine={{ stroke: '#475569' }}
            axisLine={{ stroke: '#475569' }}
            width={56}
            label={{
              value: 'T_cool RTD [°C]',
              angle: 90,
              position: 'insideRight',
              offset: 6,
              fontSize: 10,
              fill: '#fbbf24',
              fontFamily: 'monospace',
            }}
          />

          {/* ── Tooltip & Legend ─────────────────────────────────────────── */}
          <Tooltip content={<EnKFTooltip />} />
          <Legend content={<EnKFLegend />} verticalAlign="bottom" />

          {/* ── Reference lines (nominal operating point) ─────────────────── */}
          <ReferenceLine
            yAxisId="fuel"
            y={nominalFuelC}
            stroke="rgba(56,189,248,0.25)"
            strokeDasharray="8 4"
            label={{
              value: 'T_f₀',
              position: 'left',
              fontSize: 9,
              fill: 'rgba(56,189,248,0.5)',
              fontFamily: 'monospace',
            }}
          />
          <ReferenceLine
            yAxisId="cool"
            y={nominalCoolC}
            stroke="rgba(251,191,36,0.2)"
            strokeDasharray="8 4"
          />

          {/* ── CI BAND (95% confidence interval ±2σ around T_fuel estimate) ─
           *
           *  Technique: "background mask areas"
           *
           *  Layer 1 — Area ci_upper (yAxisId="fuel"):
           *    Fills from domain floor (fDomain[0]) UP TO (mean + 2σ) with
           *    a semi-transparent sky-blue.  This covers the FULL CI band AND
           *    the thin strip below it down to the axis floor.
           *
           *  Layer 2 — Area ci_lower (yAxisId="fuel"):
           *    Fills from domain floor UP TO (mean − 2σ) with the SAME colour
           *    as the chart background (CHART_BG).  This masks (hides) the
           *    lower portion of Layer 1, revealing only the CI band.
           *
           *  Result: visible blue band from (mean−2σ) to (mean+2σ), zero
           *  stacking artifacts, works at any Y-axis domain.
           * ─────────────────────────────────────────────────────────────── */}
          <Area
            yAxisId="fuel"
            type="monotone"
            dataKey="ci_upper"
            fill={C_CI_FILL}
            stroke="none"
            dot={false}
            activeDot={false}
            legendType="none"
            isAnimationActive={false}
          />
          <Area
            yAxisId="fuel"
            type="monotone"
            dataKey="ci_lower"
            fill={CHART_BG}
            stroke="none"
            dot={false}
            activeDot={false}
            legendType="none"
            isAnimationActive={false}
          />

          {/* ── Ground truth T_fuel (dashed red) — drawn BEHIND EnKF line ── */}
          <Line
            yAxisId="fuel"
            type="monotone"
            dataKey="tf_true"
            stroke={C_TRUE}
            strokeWidth={1.5}
            strokeDasharray="5 3"
            dot={false}
            activeDot={{ r: 3, strokeWidth: 0, fill: C_TRUE }}
            legendType="none"
            isAnimationActive={false}
          />

          {/* ── EnKF posterior mean T_fuel (solid cyan) ────────────────────── */}
          <Line
            yAxisId="fuel"
            type="monotone"
            dataKey="tf_mean"
            stroke={C_INFERRED}
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0, fill: C_INFERRED }}
            legendType="none"
            isAnimationActive={false}
          />

          {/* ── Noisy RTD coolant readings (amber scatter dots, no line) ───── */}
          <Line
            yAxisId="cool"
            type="monotone"
            dataKey="noisy_tc"
            stroke="none"
            strokeWidth={0}
            dot={{ r: 1.8, fill: C_NOISE, strokeWidth: 0 }}
            activeDot={{ r: 3.5, fill: C_NOISE, strokeWidth: 0 }}
            legendType="none"
            isAnimationActive={false}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
