import client from './client'
import type { RunCreate, RunResponse, RunStatusResponse, TelemetryResponse } from '../types'

/** POST /runs/ — submit a new simulation job (202 Accepted). */
export async function createRun(payload: RunCreate): Promise<RunResponse> {
  const { data } = await client.post<RunResponse>('/runs/', payload)
  return data
}

/** GET /runs/{runId}/status — poll the lifecycle state of a job. */
export async function getRunStatus(runId: string): Promise<RunStatusResponse> {
  const { data } = await client.get<RunStatusResponse>(`/runs/${runId}/status`)
  return data
}

/** GET /runs/{runId}/telemetry — fetch the full time-series output. */
export async function getRunTelemetry(runId: string): Promise<TelemetryResponse> {
  const { data } = await client.get<TelemetryResponse>(`/runs/${runId}/telemetry`)
  return data
}
