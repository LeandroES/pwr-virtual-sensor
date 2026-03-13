import client from './client'
import type {
  SensorSimulateRequest,
  SensorSimulateResponse,
  SensorJobStatus,
  SensorResultsResponse,
  SensorHistoryItem,
} from '../types'

/** POST /sensor/simulate — enqueue an EnKF virtual-sensor job. */
export async function submitSensorJob(
  payload: SensorSimulateRequest,
): Promise<SensorSimulateResponse> {
  const { data } = await client.post<SensorSimulateResponse>('/sensor/simulate', payload)
  return data
}

/** GET /sensor/{jobId}/status — lightweight lifecycle poll. */
export async function getSensorStatus(jobId: string): Promise<SensorJobStatus> {
  const { data } = await client.get<SensorJobStatus>(`/sensor/${jobId}/status`)
  return data
}

/**
 * GET /sensor/{jobId}/results — aggregate metrics + paginated chart data.
 *
 * @param maxPoints  Maximum rows to return (default 3 000).  The backend
 *                   applies SQL ROW_NUMBER() stride-based downsampling.
 */
export async function getSensorResults(
  jobId: string,
  maxPoints = 3_000,
): Promise<SensorResultsResponse> {
  const { data } = await client.get<SensorResultsResponse>(
    `/sensor/${jobId}/results`,
    { params: { max_points: maxPoints } },
  )
  return data
}

/** GET /sensor/runs/history — all simulations ordered by created_at DESC. */
export async function getSensorHistory(): Promise<SensorHistoryItem[]> {
  const { data } = await client.get<SensorHistoryItem[]>('/sensor/runs/history')
  return data
}
