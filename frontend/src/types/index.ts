// ── Virtual Sensor types ───────────────────────────────────────────────────────

export interface SensorSimulateRequest {
  external_reactivity: number          // [Δk/k]
  time_span: [number, number]          // [t_start, t_end] [s]
  dt: number                           // ≤ 0.01 s
  ensemble_size: number                // [100, 500_000]
  obs_noise_std_K: number              // RTD noise σ [K]
  enkf_obs_noise_var_K2?: number       // R [K²]; auto = σ²
  enkf_inflation_factor?: number       // [1.0, 2.0]
  device?: 'cuda' | 'cpu'
  insert_batch_size?: number
  rng_seed?: number
}

export interface SensorSimulateResponse {
  job_id: string
  status: RunStatus
  created_at: string
  ensemble_size: number
  obs_noise_std_K: number
  enkf_obs_noise_var_K2: number
  estimated_steps: number
}

export interface SensorJobStatus {
  job_id: string
  status: RunStatus
  created_at: string
  completed_at: string | null
  error_message: string | null
  external_reactivity: number
  time_span_start: number
  time_span_end: number
  dt: number
  // Smart Switch execution telemetry (null until completed)
  execution_time: number | null
  device_used: string | null
  device_reason: string | null
}

export interface SensorResultPoint {
  sim_time_s: number
  noisy_t_coolant: number        // RTD reading [K]
  inferred_t_fuel_mean: number   // EnKF estimate [K]
  inferred_t_fuel_std: number    // posterior σ [K]
  true_t_fuel: number            // ground truth [K]
  error_K: number                // inferred_mean − true [K]
}

export interface SensorMetrics {
  total_points: number
  rmse_K: number | null
  mae_K: number | null
  coverage_68pct: number | null
  coverage_95pct: number | null
  mean_ensemble_std_K: number | null
}

export interface SensorResultsResponse {
  job_id: string
  status: string
  completed_at: string | null
  error_message: string | null
  metrics: SensorMetrics | null
  total_point_count: number
  point_count: number
  truncated: boolean
  data: SensorResultPoint[]
  // Smart Switch execution telemetry (null until completed)
  execution_time: number | null
  device_used: string | null
  device_reason: string | null
}

// ── Virtual Sensor history ─────────────────────────────────────────────────────

export interface SensorHistoryItem {
  run_id: string
  created_at: string
  status: RunStatus
  device_used: string | null
  execution_time_s: number | null
  rmse_K: number | null
}

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
