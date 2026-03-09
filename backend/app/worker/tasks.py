"""
Celery tasks for the PWR Digital Twin worker.

Tasks
-----
run_pke_simulation          (original)
    Point-kinetics + TH simulation via ScipySolver.  Results stored in the
    ``telemetry`` TimescaleDB hypertable.

run_virtual_sensor_job      (new — Data Assimilation)
    Generates a ground-truth trajectory, contaminates it with severe
    Gaussian noise, then runs the EnKF step-by-step to infer the hidden
    T_fuel state.  Results are bulk-inserted into the
    ``virtual_sensor_telemetry`` hypertable using psycopg2 execute_values
    so that I/O never becomes the bottleneck while the GPU processes steps.

I/O strategy for run_virtual_sensor_job
----------------------------------------
The EnKF loop runs one step at a time on the GPU (each call to
``sensor.step_assimilation`` advances all N ensemble members via RK4 and
then applies the stochastic Kalman update — fully vectorised, no Python
loop over N).  Results are accumulated in an in-memory row buffer.

When the buffer reaches ``insert_batch_size`` rows, a single
``execute_values`` call sends them all to TimescaleDB in one round-trip.
This amortises the psycopg2 + TCP overhead over thousands of rows:

    without batching : n_steps  roundtrips  (e.g. 60 000 for a 60 s run)
    with batching    : n_steps // insert_batch_size  roundtrips  (e.g. 6)

A raw psycopg2 connection (``engine.raw_connection()``) is used for the
bulk writes to bypass SQLAlchemy's ORM overhead.  The ORM SessionLocal is
kept only for the small number of Run metadata reads/writes.
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
from celery import Task
from celery.utils.log import get_task_logger
from psycopg2.extras import execute_values
from sqlalchemy import insert

from app.core.database import SessionLocal, engine
from app.models.run import Run
from app.models.telemetry import Telemetry
from app.physics import ReactorParams, ScipySolver
from app.worker.celery_app import celery_app

logger = get_task_logger(__name__)

# ── Smart Switch threshold ────────────────────────────────────────────────────
# Below this ensemble size the PCIe H2D transfer overhead and HIP kernel-launch
# latency (~60–100 µs/dispatch) exceed the GPU compute savings.  Empirical
# crossover on RX 7900 XT (HBM2e 800 GB/s vs DDR5-6000 96 GB/s) falls between
# N = 50 000 and N = 80 000.  The conservative lower bound is used here.
# See docs/overhead_analysis.md for the full analytical model.
_GPU_ENSEMBLE_THRESHOLD: int = 50_000

# ── SQL for bulk virtual-sensor inserts ───────────────────────────────────────
# Used with psycopg2 execute_values — %s is replaced by the VALUES clause.
_INSERT_VS_SQL = """
INSERT INTO virtual_sensor_telemetry
    (ts, run_id, sim_time_s, noisy_t_coolant,
     inferred_t_fuel_mean, inferred_t_fuel_std, true_t_fuel)
VALUES %s
ON CONFLICT (ts, run_id) DO NOTHING
"""


# ── Existing task (unchanged) ─────────────────────────────────────────────────

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
    """
    db = SessionLocal()
    run_obj: Run | None = None

    try:
        run_obj = db.get(Run, uuid.UUID(run_id))
        if run_obj is None:
            logger.error("run_id=%s not found in database", run_id)
            return {"status": "error", "run_id": run_id, "points": 0}

        run_obj.status = "running"
        db.commit()
        logger.info("run_id=%s: status → running", run_id)

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


# ── Virtual Sensor task ───────────────────────────────────────────────────────

@celery_app.task(
    name="virtual_sensor.run",
    bind=True,
    acks_late=True,
    max_retries=0,
    track_started=True,
)
def run_virtual_sensor_job(
    self: Task,  # type: ignore[type-arg]
    *,
    run_id: str,
    external_reactivity: float,
    time_span: list[float],
    dt: float,
    ensemble_size: int = 10_000,
    obs_noise_std_K: float = 3.0,
    enkf_obs_noise_var_K2: float | None = None,
    enkf_inflation_factor: float = 1.02,
    device: str = "cuda",
    insert_batch_size: int = 10_000,
    rng_seed: int = 42,
) -> dict[str, object]:
    """
    End-to-end Virtual Sensor pipeline: ground truth → noise → EnKF → DB.

    Pipeline overview
    -----------------
    1. Fetch ``Run`` from DB, transition status → ``running``.
    2. GROUND TRUTH: Run ``ScipySolver`` (Radau, stiff-adaptive) on the
       nominal reactor parameters to obtain the true T_fuel and T_coolant
       trajectories at every ``dt`` interval.
    3. NOISY TELEMETRY: Add independent Gaussian noise (σ = ``obs_noise_std_K``)
       to the true T_coolant to simulate an RTD sensor with severe noise.
    4. EnKF INITIALISATION: Create an ``EnsembleSolver`` (N members on GPU)
       perturbed around the nominal parameters.  Wrap it in ``EnKFSensor``.
    5. ASSIMILATION LOOP: For each time step t[i]:
         a. solver.step(dt)  →  RK4 forecast for all N members (GPU, vectorised)
         b. Assimilate noisy_T_coolant[i]  →  stochastic EnKF update (GPU)
         c. Collect posterior statistics into ``row_buffer``
       When ``row_buffer`` reaches ``insert_batch_size``, flush via
       ``execute_values`` (one psycopg2 round-trip per batch, not per row).
    6. Final flush of any remaining rows.
    7. Transition status → ``completed``.

    Parameters
    ----------
    run_id:
        UUID string of the ``Run`` record created by the API.
    external_reactivity:
        Step reactivity insertion ρ_ext [Δk/k] applied at t=0.
    time_span:
        [t_start, t_end] in seconds.
    dt:
        Simulation / output time step [s].  Must satisfy dt ≤ 0.01 s for
        RK4 stability of the PKE system.
    ensemble_size:
        Number of EnKF ensemble members N.  Default 10 000.
        (100 000 is feasible on ≥ 8 GB VRAM; reduce if GPU OOM.)
    obs_noise_std_K:
        Standard deviation of the synthetic RTD sensor noise [K].
        Default 3.0 K — intentionally severe to stress-test the filter.
    enkf_obs_noise_var_K2:
        Observation noise variance R [K²] passed to the EnKF.  Defaults to
        ``obs_noise_std_K ** 2`` (ideal Gaussian tuning).
    enkf_inflation_factor:
        Multiplicative covariance inflation factor α ≥ 1.
        Prevents filter divergence when the ensemble spread shrinks.
    device:
        PyTorch device: ``'cuda'`` (GPU / ROCm) or ``'cpu'``.
    insert_batch_size:
        Number of rows per ``execute_values`` call.  10 000 rows ≈ 1.2 MB
        of data per round-trip — large enough to saturate Postgres throughput
        without exhausting worker RAM.
    rng_seed:
        NumPy RNG seed for reproducible sensor noise.

    Returns
    -------
    dict
        ``{"status": "completed", "run_id": ..., "points": <int>,
           "rmse_K": <float>}``
        ``rmse_K`` is the root-mean-square error of inferred vs true T_fuel.
    """
    # Lazy import: EnsembleSolver / EnKFSensor require PyTorch.
    # Placing the import here means the module still loads on workers without
    # a GPU; the error surfaces at task execution time with a clear message.
    try:
        from app.physics.assimilation import EnKFConfig, EnKFSensor
        from app.physics.tensor_solver import EnsembleSolver, NoiseConfig
    except RuntimeError as exc:
        raise RuntimeError(
            "Virtual sensor task requires PyTorch.  "
            "Install torch and ensure CUDA/ROCm is available on the worker.\n"
            f"Original error: {exc}"
        ) from exc

    # ── Smart Switch: select compute device based on ensemble size ────────────
    # The PCIe H2D transfer overhead and HIP kernel-launch latency dominate for
    # small ensembles, making CPU faster in total wall-clock time.
    # See docs/overhead_analysis.md for the full justification.
    import torch as _torch

    if ensemble_size < _GPU_ENSEMBLE_THRESHOLD:
        effective_device: str = "cpu"
        device_reason: str = (
            "Tamaño de ensamble óptimo para CPU. "
            "Previene overhead de transferencia PCIe."
        )
    else:
        effective_device = "cuda"
        device_reason = "Tamaño masivo. Aceleración GPU habilitada."

    # Hardware availability override: if GPU was selected but is not present,
    # fall back to CPU and append a diagnostic note to the reason string.
    if effective_device == "cuda" and not _torch.cuda.is_available():
        logger.warning(
            "run_id=%s: CUDA/ROCm no disponible — fallback a device='cpu'",
            run_id,
        )
        effective_device = "cpu"
        device_reason = (
            "Tamaño masivo (GPU solicitada), pero CUDA/ROCm no disponible "
            "en tiempo de ejecución. Fallback a CPU."
        )

    logger.info(
        "run_id=%s: Smart Switch → device=%s | N=%d | reason: %s",
        run_id, effective_device, ensemble_size, device_reason,
    )

    if enkf_obs_noise_var_K2 is None:
        enkf_obs_noise_var_K2 = obs_noise_std_K ** 2

    db = SessionLocal()
    run_obj: Run | None = None
    raw_conn = None

    try:
        # ── 1. Fetch Run and transition to running ────────────────────────────
        run_obj = db.get(Run, uuid.UUID(run_id))
        if run_obj is None:
            logger.error("run_id=%s not found", run_id)
            return {"status": "error", "run_id": run_id, "points": 0}

        run_obj.status = "running"
        db.commit()
        logger.info("run_id=%s [virtual_sensor]: status → running", run_id)

        anchor: datetime = run_obj.created_at
        t0, tf = float(time_span[0]), float(time_span[1])

        # ── 2. Ground truth via ScipySolver (Radau, high accuracy) ──────────
        logger.info(
            "run_id=%s: generating ground truth (ScipySolver, ρ_ext=%.2e)",
            run_id, external_reactivity,
        )
        params = ReactorParams(external_reactivity=external_reactivity)
        scipy_solver = ScipySolver()
        truth = scipy_solver.run_simulation(
            params=params,
            time_span=(t0, tf),
            dt=dt,
        )
        n_steps    = len(truth.time)
        times_np   = np.asarray(truth.time,               dtype=np.float64)
        true_tc_np = np.asarray(truth.coolant_temperature_K, dtype=np.float64)
        true_tf_np = np.asarray(truth.fuel_temperature_K,    dtype=np.float64)
        logger.info(
            "run_id=%s: ground truth ready (%d steps, dt=%.4f s)",
            run_id, n_steps, dt,
        )

        # ── 3. Synthetic noisy RTD telemetry ─────────────────────────────────
        # Add severe Gaussian noise (σ = obs_noise_std_K) to T_coolant.
        # T_fuel remains hidden — the EnKF must infer it.
        rng = np.random.default_rng(rng_seed)
        noise_np    = rng.standard_normal(n_steps) * obs_noise_std_K
        noisy_tc_np = true_tc_np + noise_np          # (n_steps,) noisy observations

        # ── 4. Initialise GPU ensemble + EnKF sensor ─────────────────────────
        # Start the simulation timer here: the ground-truth ScipySolver run
        # (step 2) is preprocessing; only the EnKF execution is timed.
        t_sim_start: float = time.perf_counter()

        logger.info(
            "run_id=%s: initialising EnsembleSolver (N=%d, device=%s)",
            run_id, ensemble_size, effective_device,
        )
        solver = EnsembleSolver(N=ensemble_size, device=effective_device)
        solver.initialize(
            params,
            noise=NoiseConfig(
                doppler_coefficient_rel=0.02,
                fuel_coolant_conductance_rel=0.02,
                nominal_power_rel=0.01,
                neutron_population_rel=0.005,
                fuel_temp_abs=2.0,
                coolant_temp_abs=1.0,
            ),
            seed=rng_seed,
        )
        sensor = EnKFSensor(
            solver,
            config=EnKFConfig(
                obs_noise_var_K2=enkf_obs_noise_var_K2,
                inflation_factor=enkf_inflation_factor,
            ),
        )
        logger.info("run_id=%s: EnKFSensor ready", run_id)

        # ── 5. Open raw psycopg2 connection for bulk inserts ─────────────────
        # engine.raw_connection() bypasses SQLAlchemy ORM to get a bare
        # psycopg2 DBAPI connection — required by execute_values.
        raw_conn = engine.raw_connection()
        cursor   = raw_conn.cursor()

        # ── 6. Assimilation loop + batched I/O ───────────────────────────────
        # Row buffer: each element is a tuple matching _INSERT_VS_SQL columns:
        #   (ts, run_id, sim_time_s, noisy_t_coolant,
        #    inferred_t_fuel_mean, inferred_t_fuel_std, true_t_fuel)
        row_buffer: list[tuple[object, ...]] = []
        run_uuid   = uuid.UUID(run_id)
        sq_errors: list[float] = []      # for RMSE computation
        rows_inserted = 0

        # ── t=0: record the prior (initial ensemble) without stepping ────────
        # The filter hasn't seen any observation yet; this captures the
        # initial uncertainty before the first predict-update cycle.
        init_mean = float(solver.state_mean[7].item())
        init_std  = float(solver.state_std[7].item())
        row_buffer.append((
            anchor + timedelta(seconds=float(times_np[0])),
            run_uuid,
            float(times_np[0]),
            float(noisy_tc_np[0]),
            init_mean,
            init_std,
            float(true_tf_np[0]),
        ))
        sq_errors.append((init_mean - true_tf_np[0]) ** 2)

        # ── t[1] … t[n_steps-1]: predict → assimilate → record ──────────────
        for i in range(1, n_steps):
            # GPU step: RK4 forecast + stochastic EnKF update
            # All N members advance simultaneously in one call — no loop over N.
            t_fuel_mean, t_fuel_var = sensor.step_assimilation(
                noisy_observation_T_coolant=float(noisy_tc_np[i]),
                dt=dt,
            )
            t_fuel_std = math.sqrt(max(t_fuel_var, 0.0))

            row_buffer.append((
                anchor + timedelta(seconds=float(times_np[i])),
                run_uuid,
                float(times_np[i]),
                float(noisy_tc_np[i]),
                t_fuel_mean,
                t_fuel_std,
                float(true_tf_np[i]),
            ))
            sq_errors.append((t_fuel_mean - true_tf_np[i]) ** 2)

            # Flush to TimescaleDB when the buffer is full.
            # execute_values sends one large VALUES clause per call —
            # one psycopg2 round-trip covers insert_batch_size rows.
            if len(row_buffer) >= insert_batch_size:
                rows_inserted += _flush_vs_rows(cursor, raw_conn, row_buffer)
                row_buffer.clear()
                logger.debug(
                    "run_id=%s: flushed batch, total inserted=%d / %d",
                    run_id, rows_inserted, n_steps,
                )

        # ── Final flush of any remaining rows ────────────────────────────────
        if row_buffer:
            rows_inserted += _flush_vs_rows(cursor, raw_conn, row_buffer)
            row_buffer.clear()

        cursor.close()
        raw_conn.close()
        raw_conn = None

        # Stop the simulation timer immediately after all I/O is complete.
        execution_time_s: float = time.perf_counter() - t_sim_start

        # ── 7. Compute RMSE and transition to completed ───────────────────────
        finite_sq = [e for e in sq_errors if math.isfinite(e)]
        rmse_K = math.sqrt(sum(finite_sq) / len(finite_sq)) if finite_sq else float('nan')
        logger.info(
            "run_id=%s: assimilation done — %d rows stored, RMSE=%.4f K, "
            "device=%s, elapsed=%.2f s",
            run_id, rows_inserted, rmse_K, effective_device, execution_time_s,
        )

        run_obj.status         = "completed"
        run_obj.completed_at   = datetime.now(timezone.utc)
        run_obj.execution_time = execution_time_s
        run_obj.device_used    = effective_device
        run_obj.device_reason  = device_reason
        db.commit()
        logger.info("run_id=%s [virtual_sensor]: status → completed", run_id)

        return {
            "status":           "completed",
            "run_id":           run_id,
            "points":           rows_inserted,
            "rmse_K":           rmse_K,
            "execution_time_s": execution_time_s,
            "device_used":      effective_device,
            "device_reason":    device_reason,
        }

    except Exception as exc:
        # Roll back ORM session
        db.rollback()
        logger.exception("run_id=%s: virtual sensor job failed: %s", run_id, exc)

        # Roll back any uncommitted raw inserts
        if raw_conn is not None:
            try:
                raw_conn.rollback()
            except Exception:
                pass
            finally:
                raw_conn.close()

        # Persist failure status
        if run_obj is not None:
            try:
                run_obj.status        = "failed"
                run_obj.error_message = str(exc)[:2000]
                run_obj.completed_at  = datetime.now(timezone.utc)
                db.commit()
            except Exception as inner:
                logger.error(
                    "run_id=%s: could not persist failure status: %s", run_id, inner
                )
                db.rollback()
        raise

    finally:
        db.close()


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _flush_vs_rows(
    cursor: object,
    raw_conn: object,
    rows: list[tuple[object, ...]],
) -> int:
    """
    Bulk-insert ``rows`` into ``virtual_sensor_telemetry`` via execute_values.

    ``execute_values`` builds a single multi-row VALUES clause and sends it in
    one psycopg2 round-trip.  ``page_size`` is set to ``len(rows)`` so the
    entire buffer goes in one statement — the caller already controls when to
    flush based on ``insert_batch_size``.

    Parameters
    ----------
    cursor:
        Open psycopg2 cursor bound to ``raw_conn``.
    raw_conn:
        Raw psycopg2 DBAPI connection (from ``engine.raw_connection()``).
    rows:
        List of 7-tuples matching the column order in ``_INSERT_VS_SQL``:
        (ts, run_id, sim_time_s, noisy_t_coolant,
         inferred_t_fuel_mean, inferred_t_fuel_std, true_t_fuel)

    Returns
    -------
    int
        Number of rows passed to this call (``len(rows)``).
    """
    execute_values(
        cursor,           # type: ignore[arg-type]
        _INSERT_VS_SQL,
        rows,
        page_size=len(rows),   # one batch = one round-trip
    )
    raw_conn.commit()           # type: ignore[union-attr]
    return len(rows)
