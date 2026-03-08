"""
FastAPI router — /sensor endpoints (Virtual Sensor / EnKF Data Assimilation).

Endpoints
---------
POST /sensor/simulate
    Validate the request, persist a Run metadata row, enqueue
    ``run_virtual_sensor_job`` as a Celery task, and return 202 Accepted.

GET /sensor/{job_id}/status
    Lightweight lifecycle poll — returns status without computing metrics
    or fetching telemetry rows.

GET /sensor/{job_id}/results
    Query ``virtual_sensor_telemetry`` for:
      • Aggregate metrics (RMSE, MAE, coverage 68/95 %) via a single SQL
        aggregate — computed over the FULL dataset regardless of pagination.
      • Paginated / downsampled time-series data for the frontend chart.

Query parameters for GET /results
----------------------------------
max_points : int  (default 5 000, max 50 000)
    Maximum number of SensorResultPoint rows to return.  When the stored
    dataset exceeds this, a stride-based subset is returned and
    ``truncated = True`` is set in the response.  Stride is computed as:
        stride = ceil(total / max_points)
    and applied via a SQL ROW_NUMBER() window function — no Python-side
    full-table load.

offset : int  (default 0)
    Row offset applied AFTER striding, enabling a second-level pagination
    on top of the downsampled dataset.

Typing / mypy
-------------
All return types are concrete (not ``Any``).  SQLAlchemy Row objects from
aggregate queries are accessed via ``.` attribute access with explicit casts.
Annotated dependencies use the ``Annotated[Session, Depends(get_db)]`` pattern.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.run import Run
from app.models.telemetry import VirtualSensorTelemetry
from app.schemas.sensor import (
    MAX_POINTS_LIMIT,
    SensorJobStatus,
    SensorMetrics,
    SensorResultPoint,
    SensorResultsResponse,
    SensorSimulateRequest,
    SensorSimulateResponse,
)
from app.worker.tasks import run_virtual_sensor_job

router = APIRouter(prefix="/sensor", tags=["virtual-sensor"])

# Typed dependency alias — identical pattern to runs.py
DbSession = Annotated[Session, Depends(get_db)]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_run_or_404(run_id: uuid.UUID, db: Session) -> Run:
    """Fetch a Run by primary key or raise HTTP 404."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {run_id} not found.",
        )
    return run


def _compute_metrics(job_id: uuid.UUID, db: Session) -> SensorMetrics | None:
    """
    Compute aggregate filter metrics in a SINGLE SQL round-trip.

    Returns ``None`` when no rows exist yet (job still in progress).

    SQL semantics
    -------------
    All aggregates use the full ``virtual_sensor_telemetry`` partition for
    ``run_id = job_id``, regardless of the ``max_points`` pagination applied
    to the data section.

    Coverage is computed with conditional sums:

        coverage_68% = 100 × COUNT(|error| ≤ 1σ) / COUNT(*)
        coverage_95% = 100 × COUNT(|error| ≤ 2σ) / COUNT(*)

    where  error = inferred_t_fuel_mean − true_t_fuel.
    """
    VST = VirtualSensorTelemetry

    # Derived expressions reused across aggregates
    diff       = VST.inferred_t_fuel_mean - VST.true_t_fuel        # signed error
    abs_diff   = func.abs(diff)
    sq_diff    = diff * diff                                         # squared error
    one_sigma  = VST.inferred_t_fuel_std
    two_sigma  = 2.0 * VST.inferred_t_fuel_std

    # Conditional indicators for coverage bands (1 if inside, 0 if outside)
    in_68 = case((abs_diff <= one_sigma, 1), else_=0)
    in_95 = case((abs_diff <= two_sigma, 1), else_=0)

    total_count = func.count()

    row = db.execute(
        select(
            total_count.label("total"),
            func.coalesce(func.sqrt(func.avg(sq_diff)), 0.0).label("rmse_K"),
            func.coalesce(func.avg(abs_diff), 0.0).label("mae_K"),
            (
                func.coalesce(func.sum(in_68), 0) * 100.0 / func.nullif(total_count, 0)
            ).label("coverage_68pct"),
            (
                func.coalesce(func.sum(in_95), 0) * 100.0 / func.nullif(total_count, 0)
            ).label("coverage_95pct"),
            func.coalesce(func.avg(one_sigma), 0.0).label("mean_ensemble_std_K"),
        ).where(VST.run_id == job_id)
    ).one()

    total: int = int(row.total)
    if total == 0:
        return None

    return SensorMetrics(
        total_points=total,
        rmse_K=float(row.rmse_K),
        mae_K=float(row.mae_K),
        coverage_68pct=float(row.coverage_68pct or 0.0),
        coverage_95pct=float(row.coverage_95pct or 0.0),
        mean_ensemble_std_K=float(row.mean_ensemble_std_K),
    )


def _fetch_strided_rows(
    job_id: uuid.UUID,
    db: Session,
    total: int,
    max_points: int,
    offset: int,
) -> list[VirtualSensorTelemetry]:
    """
    Return at most ``max_points`` rows using a SQL ROW_NUMBER() stride.

    When ``total <= max_points`` the stride is 1 (return everything).
    Otherwise ``stride = ceil(total / max_points)`` and only rows where
    ``row_number % stride == 1`` are returned.

    The ``offset`` parameter is applied as a regular SQL OFFSET after the
    stride filter — it paginates within the already-downsampled result set.

    This avoids loading the full dataset into Python memory even when
    ``total`` is in the millions.
    """
    stride = max(1, math.ceil(total / max_points))

    # ROW_NUMBER window ordered by sim_time_s — deterministic across queries
    rn = func.row_number().over(
        order_by=VirtualSensorTelemetry.sim_time_s
    ).label("_rn")

    # Subquery: annotate every row with its row number
    subq = (
        select(VirtualSensorTelemetry, rn)
        .where(VirtualSensorTelemetry.run_id == job_id)
        .subquery("ranked")
    )

    # Outer query: keep only stride-aligned rows, then paginate
    #   ROW_NUMBER starts at 1, so modulo == 1 captures rows 1, 1+stride, …
    stmt = (
        select(VirtualSensorTelemetry)
        .join(
            subq,
            VirtualSensorTelemetry.ts == subq.c.ts,
        )
        .where(
            VirtualSensorTelemetry.run_id == job_id,
            subq.c._rn % stride == 1,
        )
        .order_by(VirtualSensorTelemetry.sim_time_s)
        .offset(offset)
        .limit(max_points)
    )

    return list(db.scalars(stmt).all())


# ── POST /sensor/simulate ─────────────────────────────────────────────────────


@router.post(
    "/simulate",
    response_model=SensorSimulateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a Virtual Sensor (EnKF) job",
    description=(
        "Validates physics and filter parameters, persists a Run metadata row, "
        "and enqueues ``run_virtual_sensor_job`` as a Celery task.  "
        "Returns 202 Accepted immediately — poll GET /sensor/{job_id}/status "
        "or GET /sensor/{job_id}/results for progress."
    ),
)
def simulate_virtual_sensor(
    body: SensorSimulateRequest,
    db: DbSession,
) -> SensorSimulateResponse:
    """
    Create a virtual sensor job.

    The Celery task will:
    1. Run a ScipySolver ground-truth simulation.
    2. Add Gaussian noise (σ = obs_noise_std_K) to T_coolant.
    3. Run the EnKF step-by-step with the requested ensemble.
    4. Bulk-insert results into the ``virtual_sensor_telemetry`` hypertable.
    """
    # Resolve default R before persisting
    enkf_r: float = (
        body.enkf_obs_noise_var_K2
        if body.enkf_obs_noise_var_K2 is not None
        else body.obs_noise_std_K ** 2
    )

    # Persist Run metadata (reuses the existing runs table)
    run = Run(
        status="pending",
        created_at=datetime.now(timezone.utc),
        external_reactivity=body.external_reactivity,
        time_span_start=body.time_span[0],
        time_span_end=body.time_span[1],
        dt=body.dt,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Enqueue Celery task — all arguments must be JSON-serialisable primitives
    run_virtual_sensor_job.delay(
        run_id=str(run.id),
        external_reactivity=body.external_reactivity,
        time_span=[body.time_span[0], body.time_span[1]],
        dt=body.dt,
        ensemble_size=body.ensemble_size,
        obs_noise_std_K=body.obs_noise_std_K,
        enkf_obs_noise_var_K2=enkf_r,
        enkf_inflation_factor=body.enkf_inflation_factor,
        device=body.device,
        insert_batch_size=body.insert_batch_size,
        rng_seed=body.rng_seed,
    )

    estimated_steps = max(
        0,
        round((body.time_span[1] - body.time_span[0]) / body.dt) - 1,
    )

    return SensorSimulateResponse(
        job_id=run.id,
        status=run.status,
        created_at=run.created_at,
        ensemble_size=body.ensemble_size,
        obs_noise_std_K=body.obs_noise_std_K,
        enkf_obs_noise_var_K2=enkf_r,
        estimated_steps=estimated_steps,
    )


# ── GET /sensor/{job_id}/status ───────────────────────────────────────────────


@router.get(
    "/{job_id}/status",
    response_model=SensorJobStatus,
    summary="Poll virtual sensor job status",
    description=(
        "Lightweight endpoint for polling job lifecycle.  "
        "Does not query the telemetry hypertable.  "
        "Use GET /sensor/{job_id}/results once status is 'completed'."
    ),
)
def get_sensor_status(job_id: uuid.UUID, db: DbSession) -> SensorJobStatus:
    run = _get_run_or_404(job_id, db)
    return SensorJobStatus.model_validate(run)


# ── GET /sensor/{job_id}/results ──────────────────────────────────────────────


@router.get(
    "/{job_id}/results",
    response_model=SensorResultsResponse,
    summary="Retrieve virtual sensor results",
    description=(
        "Returns aggregate EnKF metrics (RMSE, coverage) computed over the "
        "full dataset, plus a paginated/downsampled time-series for "
        "frontend visualisation.  "
        "``metrics`` is null while the job is still in progress.  "
        "Use ``max_points`` to control chart resolution and ``offset`` to "
        "paginate within the downsampled set."
    ),
)
def get_sensor_results(
    job_id: uuid.UUID,
    db: DbSession,
    max_points: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_POINTS_LIMIT,
            description=(
                f"Maximum number of data points to return (1 – {MAX_POINTS_LIMIT}).  "
                "When the full dataset exceeds this, stride-based downsampling "
                "is applied via SQL ROW_NUMBER() — no full table load in Python."
            ),
        ),
    ] = 5_000,
    offset: Annotated[
        int,
        Query(
            ge=0,
            description=(
                "Row offset within the stride-downsampled result set.  "
                "Use together with max_points for pagination."
            ),
        ),
    ] = 0,
) -> SensorResultsResponse:
    """
    Aggregate metrics + paginated virtual-sensor time series.

    Two independent DB operations:
    1. One aggregate SQL query for SensorMetrics (full dataset, no Python loop).
    2. One strided SELECT for the data payload (at most max_points rows).
    """
    run = _get_run_or_404(job_id, db)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    # Computed regardless of run status so partial metrics are available
    # when the job is still running (the filter has committed some rows).
    metrics = _compute_metrics(job_id, db)
    total   = metrics.total_points if metrics is not None else 0

    # ── 404 guard for completely empty results ────────────────────────────────
    # If the run has failed and no rows were written, metrics is None.
    if metrics is None and run.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Job {job_id} failed before any results were written.  "
                f"Error: {run.error_message or 'unknown'}"
            ),
        )

    # ── Data payload (strided for frontend performance) ───────────────────────
    if total > 0:
        orm_rows = _fetch_strided_rows(job_id, db, total, max_points, offset)
    else:
        orm_rows = []

    data: list[SensorResultPoint] = [
        SensorResultPoint(
            sim_time_s=row.sim_time_s,
            noisy_t_coolant=row.noisy_t_coolant,
            inferred_t_fuel_mean=row.inferred_t_fuel_mean,
            inferred_t_fuel_std=row.inferred_t_fuel_std,
            true_t_fuel=row.true_t_fuel,
            error_K=row.inferred_t_fuel_mean - row.true_t_fuel,
        )
        for row in orm_rows
    ]

    truncated = total > max_points

    return SensorResultsResponse(
        job_id=job_id,
        status=run.status,
        completed_at=run.completed_at,
        error_message=run.error_message,
        metrics=metrics,
        total_point_count=total,
        point_count=len(data),
        truncated=truncated,
        data=data,
    )
