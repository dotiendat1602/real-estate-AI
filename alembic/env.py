from __future__ import annotations

import os
from logging.config import fileConfig

from dotenv import load_dotenv
from alembic import context
from sqlalchemy import engine_from_config, pool

from app.db.models import Base

# Load env vars from .env
load_dotenv()

# Alembic Config object
config = context.config

# Configure Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


def include_object(object, name, type_, reflected, compare_to):
    """
    - Chỉ migrate trong schema 'ai'
    - Bỏ qua toàn bộ objects liên quan tới langchain_* (table/index/constraint)
    """
    # 1) Nếu là table thì lọc theo schema ai + ignore langchain_*
    if type_ == "table":
        if object.schema != "ai":
            return False
        if name.startswith("langchain_"):
            return False
        return True

    # 2) Nếu là index/unique/constraint thì bỏ qua nếu parent table là langchain_*
    # object thường có .table (Index/Constraint)
    parent_table = getattr(object, "table", None)
    if parent_table is not None:
        tname = getattr(parent_table, "name", "") or ""
        tschema = getattr(parent_table, "schema", None)
        if tschema != "ai":
            return False
        if tname.startswith("langchain_"):
            return False

    # 3) Với sequence / schema objects khác: cứ để Alembic xử lý bình thường,
    # nhưng vì ta đã lọc table + index/constraint theo schema rồi nên sẽ không đụng public.
    return True


def _set_sqlalchemy_url_from_env() -> str | None:
    """
    Alembic uses sync SQLAlchemy engine. If your app uses asyncpg,
    convert DATABASE_URL from +asyncpg to +psycopg2 for migrations.
    Also escape '%' to avoid configparser interpolation errors.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None

    # Convert async driver to sync driver for Alembic
    if "+asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "+psycopg2")

    # Escape % for alembic configparser
    db_url_for_config = db_url.replace("%", "%%")

    # Override sqlalchemy.url in alembic config
    config.set_main_option("sqlalchemy.url", db_url_for_config)

    # Return real URL (not escaped) in case we want to use it directly
    return db_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no DB connection)."""
    url = _set_sqlalchemy_url_from_env() or config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
        include_object=include_object,
        version_table_schema="ai",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (with DB connection)."""
    _set_sqlalchemy_url_from_env()

    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_schemas=True,
            include_object=include_object,
            version_table_schema="ai",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()