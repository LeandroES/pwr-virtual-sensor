/**
 * VirtualSensorChart — Canvas-based EnKF Virtual Sensor chart via Apache ECharts.
 *
 * Rendering strategy:
 *   All data points from the API are passed directly to ECharts (Canvas renderer).
 *   No client-side subsampling. ECharts handles 6,001+ points natively on canvas.
 *
 * Four overlaid data series:
 *   1. CI ±2σ band  — stacked area pair [ci_lower base + ci_range delta]  (cyan, 15% opacity)
 *   2. True T_fuel  — ground truth from ScipySolver                        (coral dashed)
 *   3. Inferred T_fuel — EnKF posterior mean                               (electric cyan solid)
 *   4. Noisy RTD    — T_coolant + Gaussian noise (scatter dots)            (neon yellow)
 *
 * Dual Y-axis layout:
 *   Left  (yAxisIndex 0) — T_fuel  [°C] — inference + CI + ground truth
 *   Right (yAxisIndex 1) — T_coolant [°C] — RTD readings
 *
 * Interactivity:
 *   - dataZoom slider (bottom rail) and mouse-wheel scroll zoom on X axis
 *   - Legend click: toggle series visibility
 *   - Legend hover: focus the hovered series, blur the rest (ECharts emphasis)
 *   - Axis-aligned tooltip showing all values at the cursor time
 */
import React, { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import type { SensorResultPoint } from '../types'

// ── Corporate colour palette ──────────────────────────────────────────────────
const CHART_BG   = '#0f172a'                // slate-950
const C_INFERRED = '#00e5ff'                // electric cyan  (Material A400)
const C_TRUE     = '#ff5252'                // coral red      (Material A200)
const C_CI_FILL  = 'rgba(0,229,255,0.15)'  // cyan CI band fill
const C_NOISE    = '#ffea00'                // neon yellow    (Material A400)
const C_GRID     = 'rgba(51,65,85,0.5)'    // slate-700 translucent
const C_AXIS     = '#475569'               // slate-600
const C_TICK     = '#94a3b8'               // slate-400
const C_TEXT     = '#94a3b8'

// ── Legend series identifiers (must match series names) ───────────────────────
const S_INFERRED = 'Inferred T_fuel (EnKF)'
const S_CI       = 'CI ±2σ (95%)'
const S_TRUE     = 'True T_fuel'
const S_RTD      = 'Noisy RTD'
const S_CI_BASE  = '__ci_base__'           // internal stacking helper, hidden from legend

// ── Data transformation ───────────────────────────────────────────────────────

interface SeriesArrays {
  times:       number[]
  tfMean:      [number, number][]
  tfTrue:      [number, number][]
  noisyTc:     [number, number][]
  ciBase:      [number, number][]   // ci_lower (°C)
  ciRange:     [number, number][]   // ci_upper − ci_lower (delta, stacked on top)
  fuelMin:     number
  fuelMax:     number
  coolMin:     number
  coolMax:     number
}

function buildSeriesArrays(data: SensorResultPoint[]): SeriesArrays {
  const valid = data.filter(
    (p) => p.inferred_t_fuel_mean != null && isFinite(p.inferred_t_fuel_mean),
  )

  const tfMean:  [number, number][] = []
  const tfTrue:  [number, number][] = []
  const noisyTc: [number, number][] = []
  const ciBase:  [number, number][] = []
  const ciRange: [number, number][] = []
  const times:   number[]           = []

  let fuelMin =  Infinity
  let fuelMax = -Infinity
  let coolMin =  Infinity
  let coolMax = -Infinity

  for (const p of valid) {
    const t  = p.sim_time_s
    const mf = p.inferred_t_fuel_mean - 273.15
    const tt = p.true_t_fuel          - 273.15
    const tc = p.noisy_t_coolant      - 273.15
    const s2 = 2 * (p.inferred_t_fuel_std ?? 0)
    const lo = mf - s2
    const hi = mf + s2

    times.push(t)
    tfMean.push([t, mf])
    tfTrue.push([t, tt])
    noisyTc.push([t, tc])
    ciBase.push([t, lo])
    ciRange.push([t, hi - lo])  // stacked delta

    if (lo < fuelMin) fuelMin = lo
    if (hi > fuelMax) fuelMax = hi
    if (tt < fuelMin) fuelMin = tt
    if (tt > fuelMax) fuelMax = tt
    if (tc < coolMin) coolMin = tc
    if (tc > coolMax) coolMax = tc
  }

  const PAD = 0.5
  return {
    times,
    tfMean, tfTrue, noisyTc, ciBase, ciRange,
    fuelMin: isFinite(fuelMin) ? Math.floor(fuelMin - PAD)   : 600,
    fuelMax: isFinite(fuelMax) ? Math.ceil(fuelMax  + PAD)   : 700,
    coolMin: isFinite(coolMin) ? Math.floor(coolMin - PAD)   : 310,
    coolMax: isFinite(coolMax) ? Math.ceil(coolMax  + PAD)   : 340,
  }
}

// ── ECharts option builder ────────────────────────────────────────────────────

function buildOption(
  sa: SeriesArrays,
  nominalFuelC: number,
  nominalCoolC: number,
  labelInferred: string,
  labelCI: string,
  labelTrue: string,
  labelRTD: string,
  labelTimeAxis: string,
  labelFuelAxis: string,
  labelCoolAxis: string,
  labelT: string,
  labelTfInf: string,
  labelTfTru: string,
  labelRtdN: string,
  labelErr: string,
): EChartsOption {
  return {
    backgroundColor: CHART_BG,

    animation: false,

    // ── Grid ──────────────────────────────────────────────────────────────────
    grid: {
      top:    40,
      right:  72,
      bottom: 88,   // room for dataZoom slider
      left:   64,
    },

    // ── Legend ────────────────────────────────────────────────────────────────
    legend: {
      top:         8,
      left:        'center',
      orient:      'horizontal',
      selectedMode: true,          // click to toggle
      data: [
        { name: S_INFERRED, icon: 'path://M0,5 L28,5',
          itemStyle: { color: C_INFERRED },
          lineStyle: { color: C_INFERRED, width: 2.5 } },
        { name: S_CI,       icon: 'roundRect',
          itemStyle: { color: C_CI_FILL, borderColor: C_INFERRED, borderWidth: 1 } },
        { name: S_TRUE,     icon: 'path://M0,5 L4,5 M8,5 L12,5 M16,5 L20,5 M24,5 L28,5',
          itemStyle: { color: C_TRUE },
          lineStyle: { color: C_TRUE, width: 1.5, type: 'dashed' } },
        { name: S_RTD,      icon: 'circle',
          itemStyle: { color: C_NOISE } },
      ],
      textStyle:     { color: C_TEXT, fontSize: 12, fontFamily: 'monospace' },
      inactiveColor: '#334155',
      emphasis:      { selectorLabel: { show: true } },
    },

    // ── Axes ──────────────────────────────────────────────────────────────────
    xAxis: {
      type:  'value',
      name:  labelTimeAxis,
      nameLocation: 'middle',
      nameGap: 28,
      nameTextStyle: { color: '#64748b', fontSize: 10, fontFamily: 'monospace' },
      min: sa.times.length > 0 ? sa.times[0]  : 0,
      max: sa.times.length > 0 ? sa.times[sa.times.length - 1] : 60,
      axisLine:  { lineStyle: { color: C_AXIS } },
      axisTick:  { lineStyle: { color: C_AXIS } },
      axisLabel: {
        color:       C_TICK,
        fontSize:    10,
        fontFamily:  'monospace',
        formatter:   (v: number) => `${v.toFixed(1)}s`,
      },
      splitLine: { lineStyle: { color: C_GRID, type: 'dashed' } },
    },

    yAxis: [
      // Left — T_fuel [°C]
      {
        type:         'value',
        name:         labelFuelAxis,
        nameLocation: 'middle',
        nameGap:      44,
        nameTextStyle: { color: C_INFERRED, fontSize: 10, fontFamily: 'monospace' },
        min:    sa.fuelMin,
        max:    sa.fuelMax,
        axisLine:  { lineStyle: { color: C_AXIS } },
        axisTick:  { lineStyle: { color: C_AXIS } },
        axisLabel: {
          color: C_INFERRED, fontSize: 10, fontFamily: 'monospace',
          formatter: (v: number) => v.toFixed(1),
        },
        splitLine: { lineStyle: { color: C_GRID, type: 'dashed' } },
      },
      // Right — T_coolant [°C]
      {
        type:         'value',
        name:         labelCoolAxis,
        nameLocation: 'middle',
        nameGap:      52,
        nameTextStyle: { color: C_NOISE, fontSize: 10, fontFamily: 'monospace' },
        min:    sa.coolMin,
        max:    sa.coolMax,
        axisLine:  { lineStyle: { color: C_AXIS } },
        axisTick:  { lineStyle: { color: C_AXIS } },
        axisLabel: {
          color: C_NOISE, fontSize: 10, fontFamily: 'monospace',
          formatter: (v: number) => v.toFixed(1),
        },
        splitLine: { show: false },
      },
    ],

    // ── dataZoom — slider + mouse wheel ───────────────────────────────────────
    dataZoom: [
      {
        type:       'slider',
        xAxisIndex: 0,
        bottom:     8,
        height:     22,
        borderColor:        '#334155',
        fillerColor:        'rgba(0,229,255,0.08)',
        handleStyle:        { color: C_INFERRED, borderColor: C_INFERRED },
        moveHandleStyle:    { color: C_INFERRED },
        selectedDataBackground: {
          lineStyle: { color: C_INFERRED },
          areaStyle: { color: 'rgba(0,229,255,0.1)' },
        },
        dataBackground: {
          lineStyle: { color: '#334155' },
          areaStyle: { color: 'rgba(51,65,85,0.3)' },
        },
        textStyle: { color: C_TICK, fontSize: 9, fontFamily: 'monospace' },
        labelFormatter: (v: number) => `${Number(v).toFixed(1)}s`,
      },
      {
        type:       'inside',
        xAxisIndex: 0,
        zoomOnMouseWheel: true,
        moveOnMouseMove:  true,
      },
    ],

    // ── Tooltip ───────────────────────────────────────────────────────────────
    tooltip: {
      trigger:   'axis',
      axisPointer: {
        type:      'cross',
        lineStyle:  { color: '#475569', type: 'dashed', width: 1 },
        crossStyle: { color: '#475569', width: 1 },
      },
      backgroundColor: 'rgba(15,23,42,0.96)',
      borderColor:     '#475569',
      borderWidth:     1,
      padding:         0,
      textStyle:       { color: '#e2e8f0', fontSize: 11, fontFamily: 'monospace' },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      formatter(params: any) {
        if (!Array.isArray(params) || params.length === 0) return ''
        const t: number = params[0].axisValue ?? params[0].data?.[0]
        const byName: Record<string, number> = {}
        for (const p of params) {
          if (p.seriesName && p.data != null) {
            byName[p.seriesName] = Array.isArray(p.data) ? p.data[1] : p.value
          }
        }

        const tfm  = byName[S_INFERRED]
        const tft  = byName[S_TRUE]
        const rtd  = byName[S_RTD]
        const ciLo = byName[S_CI_BASE]

        const rows: Array<{ label: string; val: number; color: string }> = []
        if (tfm  != null) rows.push({ label: labelTfInf, val: tfm,  color: C_INFERRED })
        if (tft  != null) rows.push({ label: labelTfTru, val: tft,  color: C_TRUE     })
        if (rtd  != null) rows.push({ label: labelRtdN,  val: rtd,  color: C_NOISE    })
        if (ciLo != null && tfm != null) {
          rows.push({ label: '95% CI lo', val: ciLo,      color: '#93c5fd' })
          rows.push({ label: '95% CI hi', val: ciLo + (byName[S_CI] ?? 0), color: '#93c5fd' })
        }

        const err = tfm != null && tft != null ? tfm - tft : null

        const rowsHtml = rows.map(
          (r) =>
            `<div style="display:flex;justify-content:space-between;gap:20px">
               <span style="color:${r.color}">${r.label}</span>
               <span style="font-family:monospace;color:#e2e8f0">${r.val.toFixed(3)} °C</span>
             </div>`,
        ).join('')

        const errHtml = err != null
          ? `<div style="display:flex;justify-content:space-between;gap:20px;border-top:1px solid #334155;margin-top:4px;padding-top:4px">
               <span style="color:#94a3b8">${labelErr}</span>
               <span style="font-family:monospace;font-weight:600;color:${Math.abs(err) < 1 ? '#4ade80' : '#ff5252'}">
                 ${err >= 0 ? '+' : ''}${err.toFixed(3)} K
               </span>
             </div>`
          : ''

        return `<div style="min-width:230px;padding:0">
          <div style="padding:8px 12px 6px;border-bottom:1px solid #334155;font-weight:600;color:#67e8f9;font-family:monospace">
            ${labelT} = ${Number(t).toFixed(3)} s
          </div>
          <div style="padding:6px 12px 8px;display:flex;flex-direction:column;gap:3px">
            ${rowsHtml}${errHtml}
          </div>
        </div>`
      },
    },

    // ── Series ────────────────────────────────────────────────────────────────
    series: [
      // 1. CI base (ci_lower) — invisible line, transparent area, stacked foundation
      {
        name:        S_CI_BASE,
        type:        'line' as const,
        data:        sa.ciBase,
        yAxisIndex:  0,
        stack:       'ci_band',
        symbol:      'none',
        lineStyle:   { opacity: 0 },
        areaStyle:   { color: 'transparent', opacity: 0 },
        emphasis:    { disabled: true },
        tooltip:     { show: true },   // needed so formatter gets the ci_lower value
        legendHoverLink: false,
        silent:      true,
      },

      // 2. CI range (ci_upper − ci_lower) — stacked on top, visible fill
      {
        name:       S_CI,
        type:       'line' as const,
        data:       sa.ciRange,
        yAxisIndex: 0,
        stack:      'ci_band',
        symbol:     'none',
        lineStyle:  { opacity: 0 },
        areaStyle:  { color: C_CI_FILL, opacity: 1 },
        emphasis:   { focus: 'self' as const, areaStyle: { color: 'rgba(0,229,255,0.30)' } },
        blur:       { areaStyle: { opacity: 0.05 } },
      },

      // 3. Ground truth T_fuel (dashed coral)
      {
        name:       S_TRUE,
        type:       'line' as const,
        data:       sa.tfTrue,
        yAxisIndex: 0,
        symbol:     'none',
        lineStyle:  { color: C_TRUE, width: 1.5, type: 'dashed' as const },
        emphasis:   { focus: 'series' as const, lineStyle: { width: 2.5 } },
        blur:       { lineStyle: { opacity: 0.12 } },
        markLine: {
          silent:    true,
          symbol:    'none',
          lineStyle: { color: 'rgba(0,229,255,0.25)', type: 'dashed', width: 1 },
          label: {
            formatter: 'T_f\u2080',
            position: 'start',
            fontSize: 9,
            color: 'rgba(0,229,255,0.5)',
            fontFamily: 'monospace',
          },
          data: [{ yAxis: nominalFuelC }],
        },
      },

      // 4. EnKF posterior mean T_fuel (solid electric cyan)
      {
        name:       S_INFERRED,
        type:       'line' as const,
        data:       sa.tfMean,
        yAxisIndex: 0,
        symbol:     'none',
        lineStyle:  { color: C_INFERRED, width: 2.5 },
        emphasis:   { focus: 'series' as const, lineStyle: { width: 3.5 } },
        blur:       { lineStyle: { opacity: 0.12 } },
      },

      // 5. Noisy RTD coolant readings (neon yellow scatter dots)
      {
        name:        S_RTD,
        type:        'scatter' as const,
        data:        sa.noisyTc,
        yAxisIndex:  1,
        symbolSize:  3,
        itemStyle:   { color: C_NOISE, opacity: 0.75 },
        large:       true,             // WebGL-accelerated scatter for dense clouds
        largeThreshold: 500,
        emphasis:    { focus: 'series' as const, itemStyle: { opacity: 1 } },
        blur:        { itemStyle: { opacity: 0.06 } },
        markLine: {
          silent:    true,
          symbol:    'none',
          lineStyle: { color: 'rgba(255,234,0,0.2)', type: 'dashed', width: 1 },
          label:     { show: false },
          data: [{ yAxis: nominalCoolC }],
        },
      },
    ],
  }
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  data:          SensorResultPoint[]
  nominalFuelC:  number
  nominalCoolC:  number
}

// ── Main export ───────────────────────────────────────────────────────────────

export const VirtualSensorChart = React.forwardRef<HTMLDivElement, Props>(
  function VirtualSensorChart({ data, nominalFuelC, nominalCoolC }, ref) {
    const { t } = useTranslation()

    const sa = useMemo(() => buildSeriesArrays(data), [data])

    const option = useMemo(
      () =>
        buildOption(
          sa,
          nominalFuelC,
          nominalCoolC,
          t('legendLabels.inferred'),
          t('legendLabels.ci'),
          t('legendLabels.true'),
          t('legendLabels.rtd'),
          t('chart.timeAxis'),
          t('chart.fuelAxis'),
          t('chart.coolAxis'),
          't',
          t('chart.tfInferred'),
          t('chart.tfTrue'),
          t('chart.rtdNoisy'),
          t('chart.instantError'),
        ),
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [sa, nominalFuelC, nominalCoolC],
    )

    return (
      <div
        ref={ref}
        className="rounded-xl overflow-hidden"
        style={{ background: CHART_BG }}
      >
        <ReactECharts
          option={option}
          style={{ height: 480, width: '100%' }}
          notMerge
          lazyUpdate={false}
          opts={{ renderer: 'canvas' }}
          theme={undefined}
        />
      </div>
    )
  },
)
