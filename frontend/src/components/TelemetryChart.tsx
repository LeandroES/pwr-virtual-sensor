/**
 * TelemetryChart — dual-axis recharts visualization.
 *
 * Left  Y-axis (primary):   Thermal power in GW
 * Right Y-axis (secondary): Fuel and coolant temperatures in °C
 *
 * Uses ComposedChart so both axes share the same X domain and can be
 * cross-referenced in the unified Tooltip.
 */
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  type TooltipProps,
} from 'recharts'
import type { TelemetryPoint } from '../types'

// ── Derived chart data ────────────────────────────────────────────────────────

interface ChartRow {
  t: number         // simulation time [s]
  power_gw: number  // thermal power [GW]
  t_fuel_c: number  // fuel temperature [°C]
  t_cool_c: number  // coolant temperature [°C]
  reactivity_pcm: number
}

function toChartRows(data: TelemetryPoint[]): ChartRow[] {
  return data.map((p) => ({
    t:             Math.round(p.sim_time_s * 10) / 10,
    power_gw:      p.power_w / 1e9,
    t_fuel_c:      p.t_fuel_k - 273.15,
    t_cool_c:      p.t_coolant_k - 273.15,
    reactivity_pcm: p.reactivity * 1e5,
  }))
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-lg p-3 text-xs space-y-1.5">
      <p className="font-semibold text-gray-700 border-b pb-1 mb-1">t = {label} s</p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex justify-between gap-4">
          <span style={{ color: entry.color }} className="font-medium">{entry.name}</span>
          <span className="tabular-nums text-gray-800">{entry.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  )
}

// ── Reactivity mini-chart ─────────────────────────────────────────────────────

function ReactivityChart({ data }: { data: ChartRow[] }) {
  return (
    <div>
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
        Reactividad total ρ(t)  [pcm]
      </p>
      <ResponsiveContainer width="100%" height={120}>
        <ComposedChart data={data} margin={{ top: 4, right: 16, left: 10, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="t"
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => `${v}s`}
          />
          <YAxis
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => `${v.toFixed(1)}`}
            width={40}
          />
          <Tooltip
            formatter={(v: number) => [`${v.toFixed(3)} pcm`, 'ρ_total']}
            contentStyle={{ fontSize: 11, borderRadius: '8px' }}
          />
          <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="4 2" />
          <Line
            type="monotone"
            dataKey="reactivity_pcm"
            name="ρ (pcm)"
            stroke="#7c3aed"
            strokeWidth={1.5}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Main chart ────────────────────────────────────────────────────────────────

interface Props {
  data: TelemetryPoint[]
}

const NOMINAL_POWER_GW = 3.0
const NOMINAL_FUEL_C   = 893.0 - 273.15   // ≈ 619.85 °C
const NOMINAL_COOL_C   = 593.0 - 273.15   // ≈ 319.85 °C

export function TelemetryChart({ data }: Props) {
  const rows = toChartRows(data)

  return (
    <div className="space-y-6 fade-in">
      {/* ── Primary chart: Power + Temperatures ──────────────────────────── */}
      <div>
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Potencia térmica &amp; Temperaturas  —  respuesta transitoria
        </p>
        <ResponsiveContainer width="100%" height={380}>
          <ComposedChart
            data={rows}
            margin={{ top: 8, right: 72, left: 16, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

            <XAxis
              dataKey="t"
              type="number"
              domain={['dataMin', 'dataMax']}
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => `${v}s`}
              label={{
                value: 'Tiempo (s)',
                position: 'insideBottom',
                offset: -12,
                fontSize: 11,
                fill: '#6b7280',
              }}
            />

            {/* Left Y: Thermal power [GW] */}
            <YAxis
              yAxisId="power"
              orientation="left"
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => v.toFixed(3)}
              width={52}
              label={{
                value: 'Potencia (GW)',
                angle: -90,
                position: 'insideLeft',
                offset: -2,
                fontSize: 11,
                fill: '#2563eb',
              }}
            />

            {/* Right Y: Temperature [°C] */}
            <YAxis
              yAxisId="temp"
              orientation="right"
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => `${v.toFixed(0)}°C`}
              width={58}
              label={{
                value: 'Temperatura (°C)',
                angle: 90,
                position: 'insideRight',
                offset: 4,
                fontSize: 11,
                fill: '#dc2626',
              }}
            />

            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 12, paddingTop: '12px' }}
              iconType="plainline"
            />

            {/* Nominal reference lines */}
            <ReferenceLine
              yAxisId="power"
              y={NOMINAL_POWER_GW}
              stroke="#93c5fd"
              strokeDasharray="5 3"
              label={{ value: 'P₀', position: 'left', fontSize: 10, fill: '#93c5fd' }}
            />
            <ReferenceLine
              yAxisId="temp"
              y={NOMINAL_FUEL_C}
              stroke="#fca5a5"
              strokeDasharray="5 3"
            />
            <ReferenceLine
              yAxisId="temp"
              y={NOMINAL_COOL_C}
              stroke="#86efac"
              strokeDasharray="5 3"
            />

            {/* Data lines */}
            <Line
              yAxisId="power"
              type="monotone"
              dataKey="power_gw"
              name="Potencia (GW)"
              stroke="#2563eb"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
            <Line
              yAxisId="temp"
              type="monotone"
              dataKey="t_fuel_c"
              name="T. Combustible (°C)"
              stroke="#dc2626"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
            <Line
              yAxisId="temp"
              type="monotone"
              dataKey="t_cool_c"
              name="T. Refrigerante (°C)"
              stroke="#16a34a"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Secondary chart: Reactivity ───────────────────────────────────── */}
      <ReactivityChart data={rows} />
    </div>
  )
}
