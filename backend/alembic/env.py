"""
Alembic environment configuration.

The database URL is read from the application's Pydantic-settings object
(which pulls from the DATABASE_URL environment variable) so that migrations
run against the same database as the API and worker — no duplication in
alembic.ini required.

Autogenerate support
--------------------
All ORM models are imported below via ``app.models`` so that
``Base.metadata`` is fully populated when alembic compares against the live
schema.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Make the backend/ package importable when running alembic from any CWD ───
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Import all models so Base.metadata knows about every table ───────────────
import app.models  # noqa: F401  (side-effect: registers Run + Telemetry)

from app.core.config import settings
from app.core.database import Base

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# Override the sqlalchemy.url from alembic.ini with the live application URL
config.set_main_option("sqlalchemy.url", settings.database_url)

# Set up Python logging from alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── Migration runner helpers ──────────────────────────────────────────────────


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (emit SQL to stdout without a live DB).

    Useful for generating migration scripts to review or apply manually.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Compare server defaults so autogenerate catches DEFAULT changes
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
