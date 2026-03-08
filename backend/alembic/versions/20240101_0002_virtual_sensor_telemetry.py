"""Add virtual_sensor_telemetry hypertable for EnKF Virtual Sensor output.

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-01 01:00:00.000000 UTC

What this migration does
------------------------
1. Creates the ``virtual_sensor_telemetry`` table with columns:
     ts                    — TIMESTAMPTZ partition dimension (required first)
     run_id                — FK → runs.id  CASCADE DELETE
     sim_time_s            — elapsed simulation time [s]
     noisy_t_coolant       — RTD measurement with added Gaussian noise [K]
     inferred_t_fuel_mean  — EnKF posterior mean of hidden T_fuel [K]
     inferred_t_fuel_std   — EnKF posterior std of T_fuel [K]  (uncertainty)
     true_t_fuel           — ScipySolver ground truth T_fuel [K]  (validation)

2. Converts the table to a TimescaleDB hypertable partitioned by ``ts`` with
   a 1-day chunk interval (same pattern as ``telemetry``).

3. Creates a ``run_id`` B-Tree index so per-run queries remain fast even as
   the hypertable grows across many chunks.

TimescaleDB requirements
------------------------
- The ``timescaledb`` extension must exist (installed by migration 0001).
- The partition column ``ts`` must appear in every unique constraint, hence
  the composite primary key ``(ts, run_id)``.
- ``create_hypertable`` is idempotent via ``if_not_exists => TRUE``.

Downgrade
---------
Drops ``virtual_sensor_telemetry`` (including all hypertable chunks).
The ``runs`` table and the base ``telemetry`` hypertable are left intact.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. Create virtual_sensor_telemetry table ──────────────────────────
    # Composite PK (ts, run_id): TimescaleDB mandates the time-partition column
    # be included in every unique constraint.  Placing ts first ensures chunk
    # exclusion uses the leading index column for time-bounded queries.
    op.create_table(
        "virtual_sensor_telemetry",
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            primary_key=True,
            nullable=False,
            comment="Wall-clock UTC timestamp = run.created_at + sim_time_s",
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
            comment="Parent run identifier (FK → runs.id)",
        ),
        sa.Column(
            "sim_time_s",
            sa.Double(),
            nullable=False,
            comment="Simulation time elapsed since run start [s]",
        ),
        # ── Sensor observable ─────────────────────────────────────────────
        sa.Column(
            "noisy_t_coolant",
            sa.Double(),
            nullable=False,
            comment=(
                "Synthetic RTD measurement of T_coolant with additive "
                "Gaussian noise sigma=obs_noise_std_K [K]. "
                "This is the sole observable input to the EnKF."
            ),
        ),
        # ── EnKF posterior output ─────────────────────────────────────────
        sa.Column(
            "inferred_t_fuel_mean",
            sa.Double(),
            nullable=False,
            comment=(
                "EnKF posterior ensemble mean of the hidden fuel temperature "
                "T_f [K]. This is the Virtual Sensor's point estimate of the "
                "inaccessible fuel-cladding temperature."
            ),
        ),
        sa.Column(
            "inferred_t_fuel_std",
            sa.Double(),
            nullable=False,
            comment=(
                "EnKF posterior ensemble standard deviation of T_f [K]. "
                "Represents filter uncertainty: 68% CI = mean ± std. "
                "Shrinks as the filter converges."
            ),
        ),
        # ── Validation ground truth ───────────────────────────────────────
        sa.Column(
            "true_t_fuel",
            sa.Double(),
            nullable=False,
            comment=(
                "Ground-truth T_fuel from ScipySolver reference simulation [K]. "
                "Not available in production (fuel is hidden). Stored here "
                "to compute offline RMSE and coverage metrics."
            ),
        ),
    )

    # ── 2. B-Tree index for run_id-scoped range queries ───────────────────
    # TimescaleDB splits the table into time chunks; a standard B-Tree index
    # on run_id lets Postgres locate all chunks for a given run without
    # full-chunk scans.
    op.create_index(
        "ix_vs_telemetry_run_id",
        "virtual_sensor_telemetry",
        ["run_id"],
    )

    # ── 3. Convert to TimescaleDB hypertable ──────────────────────────────
    # chunk_time_interval = 1 day: each virtual-sensor run produces at most
    # ~86 400 s of data, so one chunk per day is a comfortable bound.
    # if_not_exists = TRUE makes this idempotent for re-runs of the migration.
    op.execute(
        """
        SELECT create_hypertable(
            'virtual_sensor_telemetry',
            'ts',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists       => TRUE
        )
        """
    )


def downgrade() -> None:
    # Dropping the hypertable also drops all its chunks automatically.
    # The run_id index is part of the table and disappears with it.
    op.drop_table("virtual_sensor_telemetry")
