"""Initial schema: runs table + telemetry TimescaleDB hypertable.

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000 UTC

Tables created
--------------
  runs        — simulation run metadata (standard PostgreSQL table)
  telemetry   — physics time-series (TimescaleDB hypertable on ``ts``)

TimescaleDB notes
-----------------
  * The ``timescaledb`` extension must be installed on the target database.
    The docker image ``timescale/timescaledb:latest-pg15`` provides it.
  * ``create_hypertable`` is called after the table is created; it converts
    the regular table into a hypertable partitioned by the ``ts`` column.
  * ``if_not_exists => TRUE`` makes the call idempotent (safe to re-run).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. Enable TimescaleDB extension (idempotent) ──────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # ── 2. runs table ─────────────────────────────────────────────────────
    op.create_table(
        "runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            comment="Unique run identifier (UUID v4)",
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="pending",
            comment="pending | running | completed | failed",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="UTC timestamp when the run was created",
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="UTC timestamp when the run finished",
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
            comment="Exception message if the run failed",
        ),
        sa.Column(
            "external_reactivity",
            sa.Double(),
            nullable=False,
            comment="Step reactivity insertion ρ_ext [Δk/k]",
        ),
        sa.Column(
            "time_span_start",
            sa.Double(),
            nullable=False,
            comment="Simulation start time [s]",
        ),
        sa.Column(
            "time_span_end",
            sa.Double(),
            nullable=False,
            comment="Simulation end time [s]",
        ),
        sa.Column(
            "dt",
            sa.Double(),
            nullable=False,
            comment="Output time step [s]",
        ),
    )
    op.create_index("ix_runs_status", "runs", ["status"])
    op.create_index("ix_runs_created_at", "runs", ["created_at"])

    # ── 3. telemetry table (will become a hypertable) ─────────────────────
    # The composite PK (ts, run_id) satisfies TimescaleDB's requirement that
    # the partition column is included in every unique constraint.
    op.create_table(
        "telemetry",
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            primary_key=True,
            nullable=False,
            comment="Wall-clock UTC = run.created_at + sim_time_s",
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
            comment="Parent run identifier",
        ),
        sa.Column(
            "sim_time_s",
            sa.Double(),
            nullable=False,
            comment="Simulation time [s]",
        ),
        sa.Column(
            "neutron_population",
            sa.Double(),
            nullable=False,
            comment="Normalized neutron population n(t) [-]",
        ),
        sa.Column(
            "power_w",
            sa.Double(),
            nullable=False,
            comment="Thermal power P(t) [W]",
        ),
        sa.Column(
            "t_fuel_k",
            sa.Double(),
            nullable=False,
            comment="Average fuel temperature [K]",
        ),
        sa.Column(
            "t_coolant_k",
            sa.Double(),
            nullable=False,
            comment="Average coolant temperature [K]",
        ),
        sa.Column(
            "reactivity",
            sa.Double(),
            nullable=False,
            comment="Total core reactivity ρ(t) [Δk/k]",
        ),
    )

    # B-Tree index for run_id-scoped queries across all hypertable chunks
    op.create_index("ix_telemetry_run_id", "telemetry", ["run_id"])

    # ── 4. Convert telemetry to a TimescaleDB hypertable ─────────────────
    # chunk_time_interval defaults to 7 days; override to 1 day since each
    # simulation run spans at most a few thousand seconds (< 1 day).
    op.execute(
        """
        SELECT create_hypertable(
            'telemetry',
            'ts',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists       => TRUE
        )
        """
    )


def downgrade() -> None:
    # Drop telemetry first (FK dependency + hypertable)
    op.drop_table("telemetry")
    op.drop_table("runs")
    # Do NOT drop the timescaledb extension — other schemas may depend on it
