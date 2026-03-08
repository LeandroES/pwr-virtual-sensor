"""
ORM model for reactor simulation time-series telemetry.

This table is registered as a **TimescaleDB hypertable** on the ``ts`` column
via the Alembic migration.  The composite primary key ``(ts, run_id)`` satisfies
TimescaleDB's requirement that every unique constraint include the partition
(time) column.

Querying pattern
----------------
  SELECT * FROM telemetry
  WHERE run_id = <uuid>
  ORDER BY sim_time_s;

The ``ix_telemetry_run_id`` B-Tree index makes this query efficient across all
hypertable chunks.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Double, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Telemetry(Base):
    __tablename__ = "telemetry"

    # ── Composite primary key (required by TimescaleDB hypertable) ────────────
    # ts MUST be first so it is the leading column used for chunk exclusion.
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
        comment="Wall-clock UTC timestamp = run.created_at + sim_time_s",
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
        comment="Parent run identifier",
    )

    # ── Simulation time ───────────────────────────────────────────────────────
    sim_time_s: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Simulation time [s] (t value from ODE solver output)",
    )

    # ── Physics state variables ───────────────────────────────────────────────
    neutron_population: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Normalized neutron population n(t) [-], n=1 at rated power",
    )
    power_w: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Thermal power P(t) = n(t) · P₀ [W]",
    )
    t_fuel_k: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Average fuel temperature T_f(t) [K]",
    )
    t_coolant_k: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Average coolant temperature T_c(t) [K]",
    )
    reactivity: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Total core reactivity ρ(t) [Δk/k]",
    )

    # ── Table-level index (for run_id lookups across all chunks) ─────────────
    __table_args__ = (
        Index("ix_telemetry_run_id", "run_id"),
    )

    def __repr__(self) -> str:
        return (
            f"Telemetry(run_id={self.run_id!s:.8}, "
            f"t={self.sim_time_s:.1f}s, P={self.power_w:.3e}W)"
        )
