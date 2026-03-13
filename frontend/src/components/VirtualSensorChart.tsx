/**
 * VirtualSensorChart — ComposedChart for the EnKF Virtual Sensor output.
 *
 * Rendering strategy:
 *   - No point reduction: all data points are rendered.
 *   - Horizontal scroll: the inner ComposedChart has a dynamic fixed pixel
 *     width (min 3 000 px, then 4 px × N points).  The outer container uses
 *     overflow-x-auto so the chart scrolls without compressing.
 *   - The custom legend is rendered OUTSIDE the scrollable area so it stays
 *     visible at all times and drives line-highlight interaction.
 *
 * Four overlaid data series (bottom → top render order):
 *
 *   1. Area  [ci_upper]   — fills from y-axis floor to +2σ bound  (cyan, 15% opacity)
 *   2. Area  [ci_lower]   — masks from y-axis floor to −2σ bound  (bg colour = invisible)
 *        ↳  Together these two areas create the ±2σ confidence band.
 *
 *   3. Line  [tf_true]    — ground-truth T_fuel from ScipySolver   (coral dashed)
 *   4. Line  [tf_mean]    — EnKF posterior mean T_fuel              (electric cyan solid)
 *   5. Line  [noisy_tc]   — noisy RTD coolant readings (dots only)  (neon yellow dots)
 *
 * Dual Y-axis layout:
 *   Left  (yAxisId="fuel")  — T_fuel range  [°C] — inference + CI + ground truth
 *   Right (yAxisId="cool")  — T_coolant range [°C] — RTD readings
 *
 * Interactivity:
 *   Hovering a legend item highlights that series and dims the rest to 0.15
 *   opacity.  Mouse-leave restores all series to full opacity.
 */
import React, { useState, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  type TooltipProps,
} from 'recharts'
import type { SensorResultPoint } from '../types'

// ── Chart background — must match the wrapper div's bg for the CI mask trick ──
const CHART_BG = '#0f172a'   // slate-950

// ── High-contrast colour palette ─────────────────────────────────────────────
const C_INFERRED = '#00e5ff'               // electric cyan (Material A400)
const C_TRUE     = '#ff5252'               // coral red     (Material A200)
const C_CI_FILL  = 'rgba(0,229,255,0.15)'  // cyan CI band fill
const C_NOISE    = '#ffea00'               // neon yellow   (Material A400)
const C_GRID     = 'rgba(51,65,85,0.5)'   // slate-700 translucent grid

// ── Legend series keys ────────────────────────────────────────────────────────
type SeriesKey = 'tf_mean' | 'ci_band' | 'tf_true' | 'noisy_tc'

// ── Derived chart shape ───────────────────────────────────────────────────────

interface ChartRow {
  t:        number   // sim_time_s [s]
  tf_mean:  number   // inferred T_fuel [°C]
  tf_true:  number   // ground-truth T_fuel [°C]
  noisy_tc: number   // noisy RTD T_coolant [°C]
  ci_upper: number   // mean + 2σ [°C]
  ci_lower: number   // mean − 2σ [°C]
}

function toChartRows(data: SensorResultPoint[]): ChartRow[] {
  return data
    .filter((p) => p.inferred_t_fuel_mean != null && isFinite(p.inferred_t_fuel_mean))
    .map((p) => {
      const twoSigma = 2 * (p.inferred_t_fuel_std ?? 0)
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

function EnKFTooltip({ active, payload, label }: TooltipProps<number, string>) {
  const { t } = useTranslation()
  if (!active || !payload || payload.length === 0) return null

  const byKey: Record<string, number> = {}
  payload.forEach((e) => { if (e.dataKey) byKey[e.dataKey as string] = e.value as number })

  const rows = [
    { name: t('chart.tfInferred'), value: byKey.tf_mean,  color: C_INFERRED, unit: '°C' },
    { name: t('chart.tfTrue'),     value: byKey.tf_true,  color: C_TRUE,     unit: '°C' },
    { name: t('chart.rtdNoisy'),   value: byKey.noisy_tc, color: C_NOISE,    unit: '°C' },
    { name: t('chart.ciUpper'),    value: byKey.ci_upper, color: '#93c5fd',  unit: '°C' },
    { name: t('chart.ciLower'),    value: byKey.ci_lower, color: '#93c5fd',  unit: '°C' },
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
            <span className="text-slate-400">{t('chart.instantError')}</span>
            <span
              className="font-mono tabular-nums font-semibold"
              style={{ color: Math.abs(error) < 1 ? '#4ade80' : '#ff5252' }}
            >
              {error >= 0 ? '+' : ''}{error.toFixed(3)} K
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Interactive legend ────────────────────────────────────────────────────────

interface LegendProps {
  hoveredLine: SeriesKey | null
  onHover: (key: SeriesKey | null) => void
}

function EnKFLegend({ hoveredLine, onHover }: LegendProps) {
  const { t } = useTranslation()

  const items: { label: string; color: string; key: SeriesKey; shape: 'line' | 'dashed' | 'area' | 'dot' }[] = [
    { label: t('legendLabels.inferred'), color: C_INFERRED, key: 'tf_mean',  shape: 'line'   },
    { label: t('legendLabels.ci'),       color: C_INFERRED, key: 'ci_band',  shape: 'area'   },
    { label: t('legendLabels.true'),     color: C_TRUE,     key: 'tf_true',  shape: 'dashed' },
    { label: t('legendLabels.rtd'),      color: C_NOISE,    key: 'noisy_tc', shape: 'dot'    },
  ]

  return (
    <div className="flex flex-wrap justify-center gap-x-6 gap-y-2 pt-3 pb-1 px-2">
      {items.map(({ label, color, key, shape }) => {
        const dimmed = hoveredLine !== null && hoveredLine !== key
        return (
          <button
            key={key}
            type="button"
            className="flex items-center gap-1.5 text-xs rounded px-1.5 py-0.5 transition-opacity
              hover:bg-slate-800 focus:outline-none cursor-default select-none"
            style={{ opacity: dimmed ? 0.35 : 1 }}
            onMouseEnter={() => onHover(key)}
            onMouseLeave={() => onHover(null)}
          >
            {shape === 'area' ? (
              <span
                className="inline-block w-8 h-3.5 rounded-sm shrink-0"
                style={{ background: C_CI_FILL, border: `1px solid ${color}` }}
              />
            ) : shape === 'dot' ? (
              <span
                className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                style={{ background: color }}
              />
            ) : (
              <svg width="28" height="10" className="shrink-0" aria-hidden>
                <line
                  x1="0" y1="5" x2="28" y2="5"
                  stroke={color}
                  strokeWidth="2"
                  strokeDasharray={shape === 'dashed' ? '5 3' : undefined}
                />
              </svg>
            )}
            <span style={{ color }}>{label}</span>
          </button>
        )
      })}
    </div>
  )
}

// ── Domain helpers ────────────────────────────────────────────────────────────

function fuelDomain(rows: ChartRow[]): [number, number] {
  if (rows.length === 0) return [600, 700]
  const lows  = rows.flatMap((r) => [r.ci_lower, r.tf_true]).filter(isFinite)
  const highs = rows.flatMap((r) => [r.ci_upper, r.tf_true]).filter(isFinite)
  if (lows.length === 0 || highs.length === 0) return [600, 700]
  const pad = 0.5
  return [Math.floor(Math.min(...lows) - pad), Math.ceil(Math.max(...highs) + pad)]
}

function coolDomain(rows: ChartRow[]): [number, number] {
  if (rows.length === 0) return [310, 340]
  const vals = rows.map((r) => r.noisy_tc).filter(isFinite)
  if (vals.length === 0) return [310, 340]
  const pad = 0.5
  return [Math.floor(Math.min(...vals) - pad), Math.ceil(Math.max(...vals) + pad)]
}

// ── Opacity helper ────────────────────────────────────────────────────────────

function seriesOpacity(key: SeriesKey, hovered: SeriesKey | null): number {
  if (!hovered || hovered === key) return 1
  return 0.15
}

// ── Main export ───────────────────────────────────────────────────────────────

interface Props {
  data:          SensorResultPoint[]
  nominalFuelC:  number
  nominalCoolC:  number
}

export const VirtualSensorChart = React.forwardRef<HTMLDivElement, Props>(
  function VirtualSensorChart({ data, nominalFuelC, nominalCoolC }, ref) {
    const { t } = useTranslation()
    const [hoveredLine, setHoveredLine] = useState<SeriesKey | null>(null)

    const rows    = useMemo(() => toChartRows(data), [data])
    const fDomain = useMemo(() => fuelDomain(rows), [rows])
    const cDomain = useMemo(() => coolDomain(rows), [rows])

    // Dynamic width: minimum 3 000 px, then 4 px per data point
    const chartWidth = Math.max(3_000, rows.length * 4)

    return (
      <div ref={ref} className="rounded-xl overflow-hidden" style={{ background: CHART_BG }}>
        {/* Scrollable chart area */}
        <div className="overflow-x-auto">
          <div style={{ width: `${chartWidth}px` }}>
            <ComposedChart
              width={chartWidth}
              height={420}
              data={rows}
              margin={{ top: 16, right: 72, left: 10, bottom: 28 }}
            >
              {/* ── Grid ──────────────────────────────────────────────────── */}
              <CartesianGrid
                strokeDasharray="2 4"
                stroke={C_GRID}
                vertical={true}
              />

              {/* ── Axes ──────────────────────────────────────────────────── */}
              <XAxis
                dataKey="t"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fontSize: 10, fill: '#94a3b8', fontFamily: 'monospace' }}
                tickFormatter={(v: number) => isFinite(v) ? `${v.toFixed(1)}s` : ''}
                tickLine={{ stroke: '#475569' }}
                axisLine={{ stroke: '#475569' }}
                label={{
                  value: t('chart.timeAxis'),
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
                tick={{ fontSize: 10, fill: C_INFERRED, fontFamily: 'monospace' }}
                tickFormatter={(v: number) => isFinite(v) ? `${v.toFixed(1)}` : ''}
                tickLine={{ stroke: '#475569' }}
                axisLine={{ stroke: '#475569' }}
                width={52}
                label={{
                  value: t('chart.fuelAxis'),
                  angle: -90,
                  position: 'insideLeft',
                  offset: 6,
                  fontSize: 10,
                  fill: C_INFERRED,
                  fontFamily: 'monospace',
                }}
              />

              {/* Right Y: T_coolant [°C] */}
              <YAxis
                yAxisId="cool"
                orientation="right"
                domain={cDomain}
                tick={{ fontSize: 10, fill: C_NOISE, fontFamily: 'monospace' }}
                tickFormatter={(v: number) => isFinite(v) ? `${v.toFixed(1)}` : ''}
                tickLine={{ stroke: '#475569' }}
                axisLine={{ stroke: '#475569' }}
                width={56}
                label={{
                  value: t('chart.coolAxis'),
                  angle: 90,
                  position: 'insideRight',
                  offset: 6,
                  fontSize: 10,
                  fill: C_NOISE,
                  fontFamily: 'monospace',
                }}
              />

              {/* ── Tooltip ──────────────────────────────────────────────── */}
              <Tooltip content={<EnKFTooltip />} />

              {/* ── Reference lines (nominal operating point) ────────────── */}
              <ReferenceLine
                yAxisId="fuel"
                y={nominalFuelC}
                stroke="rgba(0,229,255,0.25)"
                strokeDasharray="8 4"
                label={{
                  value: 'T_f₀',
                  position: 'left',
                  fontSize: 9,
                  fill: 'rgba(0,229,255,0.5)',
                  fontFamily: 'monospace',
                }}
              />
              <ReferenceLine
                yAxisId="cool"
                y={nominalCoolC}
                stroke="rgba(255,234,0,0.2)"
                strokeDasharray="8 4"
              />

              {/* ── CI BAND (95% confidence interval ±2σ) ────────────────── */}
              <Area
                yAxisId="fuel"
                type="monotone"
                dataKey="ci_upper"
                fill={C_CI_FILL}
                fillOpacity={seriesOpacity('ci_band', hoveredLine)}
                stroke="none"
                dot={false}
                activeDot={false}
                legendType="none"
                isAnimationActive={false}
              />
              {/* Mask area — keep full opacity (background fill trick) */}
              <Area
                yAxisId="fuel"
                type="monotone"
                dataKey="ci_lower"
                fill={CHART_BG}
                fillOpacity={1}
                stroke="none"
                dot={false}
                activeDot={false}
                legendType="none"
                isAnimationActive={false}
              />

              {/* ── Ground truth T_fuel (dashed coral) ───────────────────── */}
              <Line
                yAxisId="fuel"
                type="monotone"
                dataKey="tf_true"
                stroke={C_TRUE}
                strokeWidth={1.5}
                strokeDasharray="5 3"
                strokeOpacity={seriesOpacity('tf_true', hoveredLine)}
                dot={false}
                activeDot={{ r: 3, strokeWidth: 0, fill: C_TRUE }}
                legendType="none"
                isAnimationActive={false}
              />

              {/* ── EnKF posterior mean T_fuel (solid electric cyan) ──────── */}
              <Line
                yAxisId="fuel"
                type="monotone"
                dataKey="tf_mean"
                stroke={C_INFERRED}
                strokeWidth={2.5}
                strokeOpacity={seriesOpacity('tf_mean', hoveredLine)}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: C_INFERRED }}
                legendType="none"
                isAnimationActive={false}
              />

              {/* ── Noisy RTD coolant readings (neon yellow scatter dots) ─── */}
              <Line
                yAxisId="cool"
                type="monotone"
                dataKey="noisy_tc"
                stroke="none"
                strokeWidth={0}
                strokeOpacity={seriesOpacity('noisy_tc', hoveredLine)}
                dot={{ r: 1.8, fill: C_NOISE, strokeWidth: 0,
                       fillOpacity: seriesOpacity('noisy_tc', hoveredLine) }}
                activeDot={{ r: 3.5, fill: C_NOISE, strokeWidth: 0 }}
                legendType="none"
                isAnimationActive={false}
                connectNulls={false}
              />
            </ComposedChart>
          </div>
        </div>

        {/* Interactive legend — outside scroll area, always visible */}
        <EnKFLegend hoveredLine={hoveredLine} onHover={setHoveredLine} />
      </div>
    )
  },
)
