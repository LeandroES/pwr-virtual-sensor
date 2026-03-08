// ── Request payloads ──────────────────────────────────────────────────────────

export interface RunCreate {
  /** Step external reactivity insertion [Δk/k].  100 pcm = 1e-3. */
  external_reactivity: number
  /** [t_start, t_end] simulation interval [s]. */
  time_span: [number, number]
  /** Output time-step spacing [s]. */
  dt: number
}

// ── Response types (mirror backend Pydantic schemas) ──────────────────────────

export interface RunResponse {
  run_id: string
  status: RunStatus
  created_at: string
}

export type RunStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface RunStatusResponse {
  run_id: string
  status: RunStatus
  created_at: string
  completed_at: string | null
  error_message: string | null
  external_reactivity: number
  time_span_start: number
  time_span_end: number
  dt: number
}

export interface TelemetryPoint {
  sim_time_s: number
  neutron_population: number
  power_w: number
  t_fuel_k: number
  t_coolant_k: number
  reactivity: number
}

export interface TelemetryResponse {
  run_id: string
  status: string
  point_count: number
  data: TelemetryPoint[]
}
