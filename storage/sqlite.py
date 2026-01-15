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
            source_id TEXT
        )
        """
    )
    conn.commit()
