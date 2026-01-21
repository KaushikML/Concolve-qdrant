from typing import Any, Dict, List, Optional

from agents.claim_evolution_agent import ClaimEvolutionAgent
from core.utils import now_iso
from storage.agent_state import get_agent_state, set_agent_state
from storage.sqlite import get_connection


def _fetch_sources_since(last_run_ts: Optional[str]) -> List[str]:
    conn = get_connection()
    if last_run_ts:
        rows = conn.execute(
            "SELECT source_id FROM sources WHERE timestamp > ? ORDER BY timestamp",
            (last_run_ts,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT source_id FROM sources ORDER BY timestamp").fetchall()
    return [row["source_id"] for row in rows]


def run_claim_evolution_agent(
    source_ids: Optional[List[str]] = None,
    force_full_scan: bool = False,
    run_decay: bool = False,
) -> Dict[str, Any]:
    agent = ClaimEvolutionAgent()
    state = get_agent_state(agent.name)
    last_run_ts = state["last_run_ts"] if state else None

    if force_full_scan:
        source_ids = _fetch_sources_since(None)
    elif source_ids is None:
        source_ids = _fetch_sources_since(last_run_ts)

    summary = agent.run(source_ids=source_ids, force_full_scan=force_full_scan, run_decay=run_decay)
    set_agent_state(agent.name, now_iso(), None, {"last_summary": summary})
    return summary
