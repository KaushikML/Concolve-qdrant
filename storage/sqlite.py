import sqlite3
from core.config import settings


_connection = None


def get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _init_db(_connection)
    return _connection


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY,
            source_type TEXT,
            title TEXT,
            timestamp TEXT,
            url TEXT,
            text_hash TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS claim_links (
            source_id TEXT,
            claim_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            timestamp TEXT,
            claim_id TEXT,
            event_type TEXT,
            delta REAL,
            reason TEXT,
            source_id TEXT,
            agent_name TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_state (
            agent_name TEXT PRIMARY KEY,
            last_run_ts TEXT,
            cursor TEXT,
            extra_json TEXT
        )
        """
    )
    _ensure_column(conn, "events", "agent_name", "TEXT")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row["name"] for row in cursor.fetchall()]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def reset_db() -> None:
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM claim_links")
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM sources")
        conn.execute("DELETE FROM agent_state")
