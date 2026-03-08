"""
Celery tasks for the PWR Digital Twin worker.

Task: run_pke_simulation
------------------------
Receives a run_id and simulation parameters, executes the ScipySolver from
the Fase-2 physics engine, and bulk-inserts the resulting time series into
the TimescaleDB telemetry hypertable.

Lifecycle
---------
  pending  → (task picked up) → running
           → (simulation + insert complete) → completed
           → (any exception) → failed  (error_message stored on Run row)

Bulk-insert strategy
--------------------
SQLAlchemy 2.0 ``session.execute(insert(Model), list_of_dicts)`` maps to a
psycopg2 ``executemany``.  For typical simulation outputs (≤ 100 000 rows at
dt ≥ 0.01 s over 1 000 s) this is fast enough (<1 s) and avoids a dependency
on raw psycopg2 cursor APIs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from celery import Task
from celery.utils.log import get_task_logger
from sqlalchemy import insert

from app.core.database import SessionLocal
from app.models.run import Run
from app.models.telemetry import Telemetry
from app.physics import ReactorParams, ScipySolver
from app.worker.celery_app import celery_app

logger = get_task_logger(__name__)


@celery_app.task(
    name="simulation.run_pke",
    bind=True,
    acks_late=True,
    max_retries=0,
    track_started=True,
)
def run_pke_simulation(
    self: Task,  # type: ignore[type-arg]
    *,
    run_id: str,
    external_reactivity: float,
    time_span: list[float],
    dt: float,
) -> dict[str, str | int]:
    """
    Execute a point-kinetics + thermal-hydraulic simulation and persist results.

    Parameters
    ----------
    run_id:
        String representation of the ``Run.id`` UUID.
    external_reactivity:
        Step reactivity insertion ρ_ext [Δk/k].
    time_span:
        ``[t_start, t_end]`` in seconds.
    dt:
        Output time step [s].

    Returns
    -------
    dict
        ``{"status": "completed", "run_id": ..., "points": <int>}``

    Raises
    ------
    Exception
        Any exception propagates to Celery so the task is marked FAILURE.
        The ``Run.status`` is set to ``"failed"`` and ``error_message`` is
        populated before re-raising.
    """
    db = SessionLocal()
    run_obj: Run | None = None

    try:
        # ── 1. Fetch the Run record and mark it as running ────────────────
        run_obj = db.get(Run, uuid.UUID(run_id))
        if run_obj is None:
            logger.error("run_id=%s not found in database", run_id)
            return {"status": "error", "run_id": run_id, "points": 0}

        run_obj.status = "running"
        db.commit()
        logger.info("run_id=%s: status → running", run_id)

        # ── 2. Build physics parameters and run the solver ────────────────
        params = ReactorParams(external_reactivity=external_reactivity)
        solver = ScipySolver()

        result = solver.run_simulation(
            params=params,
            time_span=(time_span[0], time_span[1]),
            dt=dt,
        )
        logger.info(
            "run_id=%s: simulation finished (%d output points)",
            run_id,
            len(result.time),
        )

        # ── 3. Build bulk-insert payload ──────────────────────────────────
        # ts = wall-clock anchor + simulation elapsed seconds
        # This maps each ODE output point to a unique TIMESTAMPTZ, which is
        # required by the TimescaleDB hypertable partition dimension.
        anchor: datetime = run_obj.created_at

        rows: list[dict[str, object]] = [
            {
                "ts": anchor + timedelta(seconds=t),
                "run_id": run_obj.id,
                "sim_time_s": t,
                "neutron_population": n,
                "power_w": p,
                "t_fuel_k": tf,
                "t_coolant_k": tc,
                "reactivity": rho,
            }
            for t, n, p, tf, tc, rho in zip(
                result.time,
                result.neutron_population,
                result.power_W,
                result.fuel_temperature_K,
                result.coolant_temperature_K,
                result.reactivity,
                strict=True,
            )
        ]

        # ── 4. Bulk insert into TimescaleDB hypertable ────────────────────
        if rows:
            db.execute(insert(Telemetry), rows)

        run_obj.status = "completed"
        run_obj.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("run_id=%s: status → completed (%d points stored)", run_id, len(rows))

        return {"status": "completed", "run_id": run_id, "points": len(rows)}

    except Exception as exc:
        db.rollback()
        logger.exception("run_id=%s: simulation failed: %s", run_id, exc)

        if run_obj is not None:
            try:
                run_obj.status = "failed"
                run_obj.error_message = str(exc)[:2000]
                run_obj.completed_at = datetime.now(timezone.utc)
                db.commit()
            except Exception as inner:
                logger.error("run_id=%s: could not persist failure status: %s", run_id, inner)
                db.rollback()

        raise

    finally:
        db.close()
