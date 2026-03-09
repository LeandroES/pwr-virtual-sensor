"""
Pydantic schemas for the Virtual Sensor API (/sensor/*).

Validation philosophy
---------------------
Every numeric limit exists for a concrete physical or operational reason:

  dt ≤ 0.01 s
      RK4 stability constraint for the stiff PWR PKE system.
      The fastest eigenvalue |s₀| ≈ β/Λ ≈ 325 s⁻¹ requires
          dt ≤ 2.83 / 325 ≈ 8.7 ms  for stability.
      Exceeding this causes silent numerical blow-up, not an obvious error.

  ensemble_size ≤ 500 000
      Hard GPU-OOM guard.  At float32 precision, one (N, 9) state tensor
      occupies N × 9 × 4 B.  For N = 500 000 and 6 RK4 scratch buffers,
      total VRAM ≈ 108 MB — still safe on any modern GPU.  Beyond this the
      risk of OOM errors grows rapidly (full P_f covariance matrix, etc.).

  obs_noise_std_K ∈ [0.01, 50.0] K
      Lower bound: below 0.01 K the EnKF is essentially running a perfect
      observer — unphysical for real RTD sensors (typical σ ≈ 0.3–0.5 K).
      Upper bound: 50 K noise is already larger than the typical steady-state
      T_coolant variation in a PWR; anything larger is pathological.

  enkf_obs_noise_var_K2 ∈ (0, 2 500.0] K²  (optional)
      Corresponds to obs_noise_std_K ≤ 50 K when auto-computed.
      Deliberately allowing deliberate mis-tuning (R ≠ σ²) for research.

  enkf_inflation_factor ∈ [1.0, 2.0]
      Inflation > 2 destabilises the filter (ensemble spread becomes
      unphysical).  1.0 means no inflation.

  max_points ∈ [1, 50 000]   (GET query parameter)
      Frontend chart performance guard.  50 k points saturates any canvas
      renderer; returning more is wasteful and not user-visible.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── Hard limits (also enforced in task layer) ─────────────────────────────────
ENSEMBLE_SIZE_MAX: int      = 500_000    # GPU VRAM guard
ENSEMBLE_SIZE_MIN: int      = 100        # statistically meaningless below this
DT_MAX_S: float             = 0.01       # RK4 PKE stability ceiling [s]
OBS_NOISE_STD_MAX_K: float  = 50.0       # upper bound on RTD noise [K]
OBS_NOISE_STD_MIN_K: float  = 0.01       # lower bound on RTD noise [K]
OBS_NOISE_VAR_MAX_K2: float = OBS_NOISE_STD_MAX_K ** 2   # 2 500 K²
MAX_POINTS_LIMIT: int       = 50_000     # max rows returned to frontend
MAX_DURATION_S: float       = 86_400.0   # 24 h — same cap as RunCreate


# ── POST /sensor/simulate ─────────────────────────────────────────────────────

class SensorSimulateRequest(BaseModel):
    """
    Validated request body for POST /sensor/simulate.

    Defaults are chosen for a short, informative demo run on any hardware:
      - 60 s transient with a +50 pcm step reactivity insertion
      - 10 000 ensemble members (fits in ~30 MB VRAM at float32)
      - Severe sensor noise: σ = 3 K (6× typical RTD accuracy)
      - EnKF correctly tuned: R = σ² = 9 K²
    """

    # ── Physics ───────────────────────────────────────────────────────────────
    external_reactivity: float = Field(
        default=50e-5,           # +50 pcm step insertion
        ge=-0.10,
        le=0.10,
        description=(
            "Step external reactivity insertion ρ_ext [Δk/k].  "
            "1 pcm = 1e-5.  Bounded to ±10 % Δk/k for operational realism."
        ),
    )
    time_span: tuple[float, float] = Field(
        default=(0.0, 60.0),
        description="(t_start, t_end) integration interval [s].",
    )
    dt: float = Field(
        default=0.01,
        gt=0.0,
        le=DT_MAX_S,
        description=(
            f"Fixed integration time step [s].  Must satisfy dt ≤ {DT_MAX_S} s "
            "for RK4 stability with the stiff PWR point-kinetics system "
            "(|s₀| ≈ β/Λ ≈ 325 s⁻¹).  Recommended: 0.001–0.01 s."
        ),
    )

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble_size: int = Field(
        default=10_000,
        ge=ENSEMBLE_SIZE_MIN,
        le=ENSEMBLE_SIZE_MAX,
        description=(
            f"Number of EnKF ensemble members N.  "
            f"Range [{ENSEMBLE_SIZE_MIN}, {ENSEMBLE_SIZE_MAX}].  "
            f"Hard upper limit protects GPU VRAM: at N = {ENSEMBLE_SIZE_MAX} "
            "and float32, the state + RK4 scratch tensors occupy ~108 MB."
        ),
    )

    # ── Sensor noise ──────────────────────────────────────────────────────────
    obs_noise_std_K: float = Field(
        default=3.0,
        ge=OBS_NOISE_STD_MIN_K,
        le=OBS_NOISE_STD_MAX_K,
        description=(
            "Standard deviation σ of the synthetic RTD coolant-temperature "
            "sensor noise [K].  Added as i.i.d. Gaussian to the true T_coolant "
            f"at each step.  Range [{OBS_NOISE_STD_MIN_K}, {OBS_NOISE_STD_MAX_K}] K.  "
            "Typical RTD accuracy: 0.3–0.5 K; default 3 K is intentionally severe."
        ),
    )

    # ── EnKF tuning ───────────────────────────────────────────────────────────
    enkf_obs_noise_var_K2: float | None = Field(
        default=None,
        gt=0.0,
        le=OBS_NOISE_VAR_MAX_K2,
        description=(
            "Observation noise variance R [K²] passed to the EnKF.  "
            "Defaults to obs_noise_std_K² (ideal Gaussian tuning).  "
            "Set to a different value to study filter mis-tuning effects.  "
            f"Max: {OBS_NOISE_VAR_MAX_K2} K² (corresponds to σ = {OBS_NOISE_STD_MAX_K} K)."
        ),
    )
    enkf_inflation_factor: float = Field(
        default=1.02,
        ge=1.0,
        le=2.0,
        description=(
            "Multiplicative covariance inflation factor α ≥ 1.  "
            "Prevents ensemble spread collapse after repeated assimilation.  "
            "1.0 = disabled; 1.01–1.05 = light inflation (typical); "
            "1.10–2.0 = aggressive (use when model errors are large)."
        ),
    )

    # ── Infrastructure ────────────────────────────────────────────────────────
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description=(
            "'cuda' targets NVIDIA CUDA or AMD ROCm/HIP.  "
            "Use 'cpu' only for debugging — CPU throughput is ~100× slower "
            "for large ensembles."
        ),
    )
    insert_batch_size: int = Field(
        default=10_000,
        ge=1_000,
        le=100_000,
        description=(
            "Number of rows per psycopg2 execute_values batch insert.  "
            "Each batch is one TCP round-trip to TimescaleDB.  "
            "10 000 rows ≈ 1.2 MB/batch — optimal for most network paths."
        ),
    )
    rng_seed: int = Field(
        default=42,
        description="NumPy RNG seed for reproducible synthetic sensor noise.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("time_span", mode="before")
    @classmethod
    def coerce_time_span(cls, v: object) -> tuple[float, float]:
        """Accept JSON array or Python tuple; coerce elements to float."""
        if isinstance(v, (list, tuple)) and len(v) == 2:  # type: ignore[arg-type]
            return (float(v[0]), float(v[1]))  # type: ignore[index]
        raise ValueError("time_span must be a two-element array [t_start, t_end]")

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "SensorSimulateRequest":
        """Cross-field consistency checks."""
        t0, tf = self.time_span

        if tf <= t0:
            raise ValueError(
                f"time_span[1] ({tf}) must be strictly greater than "
                f"time_span[0] ({t0})"
            )

        duration = tf - t0
        if duration > MAX_DURATION_S:
            raise ValueError(
                f"Simulation duration {duration:.0f} s exceeds the maximum "
                f"allowed {MAX_DURATION_S:.0f} s (24 h)"
            )

        min_steps = 10
        n_steps = int(round(duration / self.dt))
        if n_steps < min_steps:
            raise ValueError(
                f"Simulation must produce at least {min_steps} integration "
                f"steps; got {n_steps} (duration={duration} s, dt={self.dt} s).  "
                "Increase the duration or decrease dt."
            )

        # Resolve obs noise variance
        if self.enkf_obs_noise_var_K2 is None:
            self.enkf_obs_noise_var_K2 = self.obs_noise_std_K ** 2

        return self


class SensorSimulateResponse(BaseModel):
    """
    Payload returned immediately by POST /sensor/simulate (202 Accepted).

    The ``job_id`` is used to poll results via GET /sensor/{job_id}/results.
    """

    job_id: uuid.UUID = Field(description="Unique job identifier — use for GET /results")
    status: str        = Field(description="Always 'pending' at creation time")
    created_at: datetime

    # Echo the key EnKF parameters back so the client can display them
    # without a second round-trip.
    ensemble_size: int             = Field(description="Ensemble member count N")
    obs_noise_std_K: float         = Field(description="Synthetic sensor noise σ [K]")
    enkf_obs_noise_var_K2: float   = Field(description="EnKF observation noise variance R [K²]")
    estimated_steps: int           = Field(description="Number of assimilation steps (n_steps − 1)")


# ── GET /sensor/{job_id}/results ──────────────────────────────────────────────

class SensorResultPoint(BaseModel):
    """
    One time-series sample from the virtual sensor output.

    All temperatures in Kelvin.  ``error_K`` is a derived field:
        error_K = inferred_t_fuel_mean − true_t_fuel
    A positive error means the filter over-estimated T_fuel.
    The 68% confidence interval is [mean − std, mean + std].
    """

    sim_time_s:           float = Field(description="Simulation time [s]")
    noisy_t_coolant:      float = Field(description="RTD measurement of T_coolant with noise [K]")
    inferred_t_fuel_mean: float = Field(description="EnKF posterior mean T_fuel [K]  — virtual sensor output")
    inferred_t_fuel_std:  float = Field(description="EnKF posterior std T_fuel [K]  — uncertainty (1σ)")
    true_t_fuel:          float = Field(description="Ground-truth T_fuel from ScipySolver [K]  — validation only")
    error_K:              float = Field(description="Signed inference error: inferred_mean − true [K]")


class SensorMetrics(BaseModel):
    """
    Aggregate statistics computed over the full virtual sensor run.

    All metrics are computed at the TimescaleDB layer (one SQL aggregate
    round-trip) so they reflect the complete dataset, not just the paginated
    subset returned in ``data``.
    """

    total_points:        int   = Field(description="Total rows in the hypertable for this job")
    rmse_K:              float = Field(description="Root-mean-square error |inferred − true| [K]")
    mae_K:               float = Field(description="Mean absolute error [K]")
    coverage_68pct:      float = Field(
        description=(
            "Empirical 68% coverage: fraction of steps where "
            "|error| ≤ 1·inferred_std [%].  "
            "A well-calibrated filter gives ≈ 68%."
        )
    )
    coverage_95pct:      float = Field(
        description=(
            "Empirical 95% coverage: fraction of steps where "
            "|error| ≤ 2·inferred_std [%].  "
            "A well-calibrated filter gives ≈ 95%."
        )
    )
    mean_ensemble_std_K: float = Field(
        description="Time-average of the posterior uncertainty (mean std over all steps) [K]"
    )


class SensorResultsResponse(BaseModel):
    """
    Response body for GET /sensor/{job_id}/results.

    When the run is still ``pending`` or ``running``, ``metrics`` is ``None``
    and ``data`` contains whatever rows have been committed so far.
    When ``truncated`` is True, only the first ``point_count`` rows are
    returned; use the ``offset`` query parameter to paginate.
    """

    job_id:            uuid.UUID
    status:            str
    completed_at:      datetime | None = None
    error_message:     str | None      = None

    metrics:           SensorMetrics | None = Field(
        default=None,
        description="Aggregate statistics (null while the job is in progress)",
    )

    total_point_count: int  = Field(description="Total rows stored in TimescaleDB")
    point_count:       int  = Field(description="Rows returned in this response")
    truncated:         bool = Field(
        description=(
            "True when total_point_count > max_points query parameter.  "
            "Use 'offset' to retrieve subsequent pages."
        )
    )
    data:              list[SensorResultPoint]

    # Smart Switch execution telemetry (null while the job is in progress)
    execution_time: float | None = Field(
        default=None,
        description=(
            "Wall-clock duration of the EnKF simulation phase [s], measured "
            "from EnsembleSolver initialisation to final DB flush.  "
            "Null while the job is pending or running."
        ),
    )
    device_used: str | None = Field(
        default=None,
        description=(
            "Compute device selected by the Smart Switch: 'cuda' or 'cpu'.  "
            "Determined by the ensemble-size threshold (N < 50 000 → 'cpu'; "
            "N ≥ 50 000 → 'cuda'), with a hardware availability override."
        ),
    )
    device_reason: str | None = Field(
        default=None,
        description=(
            "Human-readable explanation for the device selection decision.  "
            "Populated on completion."
        ),
    )


# ── GET /sensor/{job_id}/status (lightweight poll) ────────────────────────────

class SensorJobStatus(BaseModel):
    """
    Lightweight status response for polling before the run completes.

    Avoids the overhead of computing metrics or fetching telemetry rows.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    job_id:        uuid.UUID      = Field(alias="id")
    status:        str
    created_at:    datetime
    completed_at:  datetime | None = None
    error_message: str | None      = None

    # Echo simulation parameters (stored on the Run row)
    external_reactivity: float
    time_span_start:     float
    time_span_end:       float
    dt:                  float

    # Smart Switch execution telemetry (null until the job completes)
    execution_time: float | None = Field(
        default=None,
        description=(
            "Wall-clock duration of the EnKF simulation phase [s], measured "
            "from EnsembleSolver initialisation to final DB flush.  "
            "Null while the job is pending or running."
        ),
    )
    device_used: str | None = Field(
        default=None,
        description=(
            "Compute device selected by the Smart Switch: 'cuda' or 'cpu'.  "
            "Null while the job is pending."
        ),
    )
    device_reason: str | None = Field(
        default=None,
        description=(
            "Human-readable explanation for the device selection decision.  "
            "Null while the job is pending."
        ),
    )
