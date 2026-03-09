"""Add execution_time, device_used, device_reason columns to runs table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-09 00:00:00.000000 UTC

What this migration does
------------------------
Adds three nullable columns to the ``runs`` table to support the Smart Switch
hardware selector and simulation timing instrumentation introduced in the
``run_virtual_sensor_job`` Celery task:

  execution_time  FLOAT
      Wall-clock duration of the simulation phase in seconds, measured with
      ``time.perf_counter()`` and covering the period from EnsembleSolver
      initialisation through the final psycopg2 flush.  NULL until the job
      reaches the 'completed' state.

  device_used  VARCHAR(16)
      Compute device selected by the Smart Switch: 'cuda' or 'cpu'.
      Determined by the ensemble-size threshold (N < 50 000 → 'cpu';
      N >= 50 000 → 'cuda'), with a hardware availability override.
      NULL until the job starts execution.

  device_reason  TEXT
      Human-readable explanation for the device selection decision, suitable
      for display in the frontend and operational dashboards.  NULL until the
      job starts execution.

All three columns are nullable so that existing rows (created before this
migration) and the ``run_pke_simulation`` task (which does not populate them)
remain valid without a data backfill.

Upgrade/downgrade
-----------------
  upgrade  : ADD COLUMN for each of the three columns
  downgrade: DROP COLUMN for each of the three columns (data is lost)
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# Alembic revision identifiers
revision: str      = "0003"
down_revision: str = "0002"
branch_labels       = None
depends_on          = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "execution_time",
            sa.Float(),
            nullable=True,
            comment="Wall-clock duration of the simulation phase [s] (perf_counter)",
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "device_used",
            sa.String(length=16),
            nullable=True,
            comment="Compute device selected by the smart switch: 'cuda' or 'cpu'",
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "device_reason",
            sa.Text(),
            nullable=True,
            comment="Human-readable explanation for the device selection decision",
        ),
    )


def downgrade() -> None:
    op.drop_column("runs", "device_reason")
    op.drop_column("runs", "device_used")
    op.drop_column("runs", "execution_time")
