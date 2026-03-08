"""
ORM model for simulation run metadata.

Each row represents one physics simulation job — its parameters, lifecycle
status, and timing.  Telemetry (time-series output) lives in a separate
hypertable linked to this table via run_id.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Double, String, Text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Run(Base):
    __tablename__ = "runs"

    # ── Identity ──────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique run identifier (UUID v4)",
    )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        comment="pending | running | completed | failed",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp when the run was created",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp when the run finished (completed or failed)",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Exception message if the run failed",
    )

    # ── Simulation parameters (stored for reproducibility) ────────────────────
    external_reactivity: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Step external reactivity insertion ρ_ext [Δk/k]",
    )
    time_span_start: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Simulation start time [s]",
    )
    time_span_end: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Simulation end time [s]",
    )
    dt: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        comment="Output time step [s]",
    )

    def __repr__(self) -> str:
        return (
            f"Run(id={self.id!s:.8}, status={self.status!r}, "
            f"ρ_ext={self.external_reactivity:.2e})"
        )
