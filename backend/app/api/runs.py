"""
FastAPI router — /runs endpoints.

POST /runs/
    Validate the request, persist a Run metadata row, dispatch the Celery
    task, and return a 202 Accepted response with the job ID.

GET /runs/{run_id}/status
    Return the current lifecycle state of a run (pending / running /
    completed / failed) plus timing and parameter metadata.

GET /runs/{run_id}/telemetry
    Return the full time-series telemetry for a completed run, ordered by
    simulation time.  Returns an empty ``data`` list if the run has not yet
    produced any results.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.run import Run
from app.models.telemetry import Telemetry
from app.schemas.runs import (
    RunCreate,
    RunResponse,
    RunStatusResponse,
    TelemetryPoint,
    TelemetryResponse,
)
from app.worker.tasks import run_pke_simulation

router = APIRouter(prefix="/runs", tags=["runs"])

# Typed dependency alias — cleaner than repeating Depends(get_db) everywhere
DbSession = Annotated[Session, Depends(get_db)]


def _get_run_or_404(run_id: uuid.UUID, db: Session) -> Run:
    """Fetch a Run by primary key or raise 404."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )
    return run


# ── POST /runs/ ───────────────────────────────────────────────────────────────


@router.post(
    "/",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new simulation run",
    description=(
        "Validates the request payload, persists a Run record with status "
        "``pending``, enqueues the PKE simulation as a Celery task, and "
        "returns the job ID immediately (202 Accepted)."
    ),
)
def create_run(body: RunCreate, db: DbSession) -> RunResponse:
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

    # Dispatch to Celery — task receives JSON-serializable primitives only
    run_pke_simulation.delay(
        run_id=str(run.id),
        external_reactivity=body.external_reactivity,
        time_span=[body.time_span[0], body.time_span[1]],
        dt=body.dt,
    )

    return RunResponse(
        run_id=run.id,
        status=run.status,
        created_at=run.created_at,
    )


# ── GET /runs/{run_id}/status ─────────────────────────────────────────────────


@router.get(
    "/{run_id}/status",
    response_model=RunStatusResponse,
    summary="Get run lifecycle status",
)
def get_run_status(run_id: uuid.UUID, db: DbSession) -> RunStatusResponse:
    run = _get_run_or_404(run_id, db)
    return RunStatusResponse.model_validate(run)


# ── GET /runs/{run_id}/telemetry ──────────────────────────────────────────────


@router.get(
    "/{run_id}/telemetry",
    response_model=TelemetryResponse,
    summary="Retrieve simulation telemetry",
    description=(
        "Returns all stored time-series points for the run, ordered by "
        "``sim_time_s``.  If the run is still in progress the ``data`` list "
        "will be empty or partially populated."
    ),
)
def get_telemetry(run_id: uuid.UUID, db: DbSession) -> TelemetryResponse:
    run = _get_run_or_404(run_id, db)

    rows = (
        db.execute(
            select(Telemetry)
            .where(Telemetry.run_id == run_id)
            .order_by(Telemetry.sim_time_s)
        )
        .scalars()
        .all()
    )

    return TelemetryResponse(
        run_id=run_id,
        status=run.status,
        point_count=len(rows),
        data=[TelemetryPoint.model_validate(row) for row in rows],
    )
