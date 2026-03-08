"""
Pydantic schemas for the /runs API endpoints.

Separation of concerns
----------------------
  RunCreate         — validated request body for POST /runs/
  RunResponse       — minimal response immediately after job creation
  RunStatusResponse — full lifecycle metadata for GET /runs/{id}/status
  TelemetryPoint    — one time-series row for GET /runs/{id}/telemetry
  TelemetryResponse — paginated telemetry payload
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RunCreate(BaseModel):
    """
    Validated request body for POST /runs/.

    All physics parameters default to their nominal PWR values so a minimal
    call body of ``{}`` triggers a steady-state (zero-perturbation) run.
    """

    external_reactivity: float = Field(
        default=0.0,
        ge=-0.10,
        le=0.10,
        description=(
            "Step external reactivity insertion ρ_ext [Δk/k].  "
            "100 pcm = 1e-3.  Bounded to ±10 % Δk/k for safety."
        ),
    )
    time_span: tuple[float, float] = Field(
        default=(0.0, 600.0),
        description="(t_start, t_end) integration interval [s]",
    )
    dt: float = Field(
        default=1.0,
        gt=0.0,
        le=60.0,
        description="Output time-step spacing [s].  Must be positive and ≤ 60 s.",
    )

    @field_validator("time_span", mode="before")
    @classmethod
    def coerce_time_span(cls, v: object) -> tuple[float, float]:
        """Accept list or tuple and convert to tuple[float, float]."""
        if isinstance(v, (list, tuple)) and len(v) == 2:  # type: ignore[arg-type]
            return (float(v[0]), float(v[1]))  # type: ignore[index]
        raise ValueError("time_span must be a two-element array [t_start, t_end]")

    @model_validator(mode="after")
    def check_span_and_dt(self) -> "RunCreate":
        t0, tf = self.time_span
        if tf <= t0:
            raise ValueError("time_span[1] must be strictly greater than time_span[0]")
        if (tf - t0) > 86_400:
            raise ValueError("Simulation duration cannot exceed 86 400 s (24 h)")
        if self.dt > (tf - t0):
            raise ValueError("dt must not exceed the total simulation duration")
        return self


class RunResponse(BaseModel):
    """Minimal payload returned immediately after POST /runs/."""

    run_id: uuid.UUID = Field(description="Unique job identifier")
    status: str = Field(description="Initial lifecycle status (always 'pending')")
    created_at: datetime = Field(description="UTC timestamp of job creation")


class RunStatusResponse(BaseModel):
    """Full lifecycle metadata returned by GET /runs/{id}/status."""

    model_config = ConfigDict(from_attributes=True)

    run_id: uuid.UUID = Field(alias="id", description="Unique job identifier")
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None

    # Echo back the simulation parameters for convenience
    external_reactivity: float
    time_span_start: float
    time_span_end: float
    dt: float

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class TelemetryPoint(BaseModel):
    """Single time-series sample from the ODE solution."""

    model_config = ConfigDict(from_attributes=True)

    sim_time_s: float = Field(description="Simulation time [s]")
    neutron_population: float = Field(description="Normalized neutron population n(t) [-]")
    power_w: float = Field(description="Thermal power P(t) [W]")
    t_fuel_k: float = Field(description="Average fuel temperature T_f(t) [K]")
    t_coolant_k: float = Field(description="Average coolant temperature T_c(t) [K]")
    reactivity: float = Field(description="Total core reactivity ρ(t) [Δk/k]")


class TelemetryResponse(BaseModel):
    """Full telemetry payload returned by GET /runs/{id}/telemetry."""

    run_id: uuid.UUID
    status: str
    point_count: int = Field(description="Number of time-series points stored")
    data: list[TelemetryPoint]
