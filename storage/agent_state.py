import json
from typing import Any, Dict, Optional

from storage.sqlite import get_connection


def init_agent_state_table() -> None:
    conn = get_connection()
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
    conn.commit()


def get_agent_state(agent_name: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT agent_name, last_run_ts, cursor, extra_json FROM agent_state WHERE agent_name = ?",
        (agent_name,),
    ).fetchone()
    if not row:
        return None
    extra_json = row["extra_json"]
    extra = None
    if extra_json:
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError:
            extra = extra_json
    return {
        "agent_name": row["agent_name"],
        "last_run_ts": row["last_run_ts"],
        "cursor": row["cursor"],
        "extra_json": extra,
    }


def set_agent_state(
    agent_name: str,
    last_run_ts: Optional[str],
    cursor: Optional[str],
    extra_json: Optional[Any],
) -> None:
    conn = get_connection()
    payload = None
    if extra_json is not None:
        payload = extra_json if isinstance(extra_json, str) else json.dumps(extra_json)
    statement = """
        INSERT INTO agent_state (agent_name, last_run_ts, cursor, extra_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(agent_name) DO UPDATE SET
            last_run_ts=excluded.last_run_ts,
            cursor=excluded.cursor,
            extra_json=excluded.extra_json
        """
    params = (agent_name, last_run_ts, cursor, payload)
    if conn.in_transaction:
        conn.execute(statement, params)
    else:
        with conn:
            conn.execute(statement, params)
