"""
DB access (blueprint §14). Phase 2: use SQLAlchemy or plain psycopg over the
schema in schema.sql. Kept minimal here so the scaffold has no heavy ORM opinion
baked in before you need one.
"""
from __future__ import annotations

from pathlib import Path

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


def schema_ddl() -> str:
    """Return the DDL string (apply on startup / in a migration)."""
    return SCHEMA_SQL.read_text(encoding="utf-8")
