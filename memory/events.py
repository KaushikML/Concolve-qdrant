from typing import Optional

from core.utils import now_iso
from storage.sqlite import get_connection


def log_event(
    claim_id: str,
    event_type: str,
    delta: float,
    reason: str,
    source_id: Optional[str] = None,
) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO events (timestamp, claim_id, event_type, delta, reason, source_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now_iso(), claim_id, event_type, delta, reason, source_id),
        )
