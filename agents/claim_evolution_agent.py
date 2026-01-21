from typing import Any, Dict, List, Optional, Set, Tuple

from qdrant_client.http import models

from agents.base_agent import BaseAgent
from agents.utils import (
    CONTRADICTION_THRESHOLD,
    CONTRADICTION_WINDOW_DAYS,
    TREND_WINDOW_DAYS,
    VOLATILITY_ALERT_THRESHOLD,
    VOLATILITY_WINDOW_DAYS,
    chunk_list,
    compute_alert_level,
    contradiction_ratio,
    cutoff_days,
    is_within_days,
    safe_float,
    safe_int,
    volatility_score,
)
from core.utils import now_iso
from memory.decay import apply_decay
from memory.events import log_event
from models.stance_classifier import classify_stance
from qdrant_store.client import get_client
from qdrant_store.collections import CLAIMS_COLLECTION, EVIDENCE_COLLECTION, MEDIA_COLLECTION
from qdrant_store.crud import get_point, update_payload
from storage.sqlite import get_connection


class ClaimEvolutionAgent(BaseAgent):
    name = "claim_evolution"

    def run(
        self,
        source_ids: Optional[List[str]] = None,
        force_full_scan: bool = False,
        run_decay: bool = False,
    ) -> Dict[str, Any]:
        summary = {
            "claims_processed": 0,
            "claims_updated": 0,
            "claims_disputed": 0,
            "high_alerts": 0,
            "medium_alerts": 0,
            "volatility_flags": 0,
        }

        if run_decay:
            updated = apply_decay()
            log_event(
                "system",
                "agent_decay_run",
                float(updated),
                f"decay applied to {updated} claims",
                agent_name=self.name,
            )

        claim_ids = self._fetch_claim_ids(source_ids or [], force_full_scan)
        if not claim_ids:
            return summary

        trend_counts = self._fetch_trend_counts(claim_ids)
        conn = get_connection()

        for claim_id in claim_ids:
            point = get_point(CLAIMS_COLLECTION, claim_id)
            if not point or not point.payload:
                continue
            payload = point.payload

            trend_score = float(trend_counts.get(claim_id, 0))
            support_recent, contradict_recent = self._evidence_stance_counts(
                claim_id, payload.get("claim_text", "")
            )
            ratio = contradiction_ratio(support_recent, contradict_recent)
            meme_variants = self._meme_variant_count(payload.get("linked_media_ids", []))
            vol_events = self._confidence_event_count(conn, claim_id)
            vol_score = volatility_score(vol_events)
            alert_level = compute_alert_level(trend_score, ratio, vol_score)

            updates: Dict[str, Any] = {
                "trend_score": trend_score,
                "contradiction_ratio": ratio,
                "meme_variant_count": meme_variants,
                "volatility_score": vol_score,
                "alert_level": alert_level,
                "last_agent_update_ts": now_iso(),
            }

            previous_trend = safe_float(payload.get("trend_score"))
            if trend_score > previous_trend:
                log_event(
                    claim_id,
                    "agent_reinforce",
                    trend_score - previous_trend,
                    f"trend window mentions={trend_score}",
                    agent_name=self.name,
                )

            previous_ratio = safe_float(payload.get("contradiction_ratio"))
            if abs(ratio - previous_ratio) >= 0.1:
                log_event(
                    claim_id,
                    "agent_contradict_shift",
                    ratio - previous_ratio,
                    f"support={support_recent} contradict={contradict_recent}",
                    agent_name=self.name,
                )

            if ratio >= CONTRADICTION_THRESHOLD and payload.get("status") != "disputed":
                updates["status"] = "disputed"
                log_event(
                    claim_id,
                    "agent_status_update",
                    1.0,
                    "status set to disputed by contradiction ratio",
                    agent_name=self.name,
                )

            previous_alert = payload.get("alert_level")
            if previous_alert != alert_level and alert_level in {"medium", "high"}:
                log_event(
                    claim_id,
                    "agent_trend_alert",
                    0.0,
                    f"alert level={alert_level} trend={trend_score} ratio={round(ratio, 3)}",
                    agent_name=self.name,
                )

            previous_vol = safe_float(payload.get("volatility_score"))
            if vol_score >= VOLATILITY_ALERT_THRESHOLD and previous_vol < VOLATILITY_ALERT_THRESHOLD:
                summary["volatility_flags"] += 1
                log_event(
                    claim_id,
                    "agent_volatility",
                    vol_score - previous_vol,
                    f"confidence events={vol_events}",
                    agent_name=self.name,
                )

            update_payload(CLAIMS_COLLECTION, claim_id, updates)
            summary["claims_updated"] += 1
            summary["claims_processed"] += 1
            if alert_level == "high":
                summary["high_alerts"] += 1
            elif alert_level == "medium":
                summary["medium_alerts"] += 1
            if updates.get("status") == "disputed":
                summary["claims_disputed"] += 1

        return summary

    def _fetch_claim_ids(self, source_ids: List[str], force_full_scan: bool) -> List[str]:
        conn = get_connection()
        if force_full_scan:
            rows = conn.execute("SELECT DISTINCT claim_id FROM claim_links").fetchall()
            return list({row["claim_id"] for row in rows})
        if not source_ids:
            return []
        claim_ids: List[str] = []
        for batch in chunk_list(source_ids):
            placeholders = ",".join("?" for _ in batch)
            statement = f"""
                SELECT DISTINCT claim_id FROM claim_links
                WHERE source_id IN ({placeholders})
                """
            rows = conn.execute(statement, tuple(batch)).fetchall()
            claim_ids.extend([row["claim_id"] for row in rows])
        return list({cid for cid in claim_ids if cid})

    def _fetch_trend_counts(self, claim_ids: List[str]) -> Dict[str, int]:
        conn = get_connection()
        cutoff = cutoff_days(TREND_WINDOW_DAYS).isoformat() + "Z"
        counts: Dict[str, int] = {}
        for batch in chunk_list(claim_ids):
            placeholders = ",".join("?" for _ in batch)
            statement = f"""
                SELECT claim_links.claim_id AS claim_id, COUNT(DISTINCT sources.source_id) AS cnt
                FROM claim_links
                JOIN sources ON sources.source_id = claim_links.source_id
                WHERE sources.timestamp >= ? AND claim_links.claim_id IN ({placeholders})
                GROUP BY claim_links.claim_id
                """
            params = (cutoff, *batch)
            rows = conn.execute(statement, params).fetchall()
            for row in rows:
                counts[row["claim_id"]] = int(row["cnt"])
        return counts

    def _evidence_stance_counts(self, claim_id: str, claim_text: str) -> Tuple[int, int]:
        support = 0
        contradict = 0
        client = get_client()
        offset = None
        query_filter = models.Filter(
            must=[models.FieldCondition(key="claim_id", match=models.MatchValue(value=claim_id))]
        )
        while True:
            points, next_offset = client.scroll(
                collection_name=EVIDENCE_COLLECTION,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                if not is_within_days(payload.get("timestamp"), CONTRADICTION_WINDOW_DAYS):
                    continue
                stance = payload.get("stance")
                if not stance and claim_text:
                    snippet = payload.get("snippet_text", "")
                    stance = classify_stance(snippet, claim_text)
                    update_payload(EVIDENCE_COLLECTION, str(point.id), {"stance": stance})
                if stance == "support":
                    support += 1
                elif stance == "contradict":
                    contradict += 1
            if next_offset is None:
                break
            offset = next_offset
        return support, contradict

    def _meme_variant_count(self, linked_media_ids: Optional[List[str]]) -> int:
        if not linked_media_ids:
            return 0
        phashes: Set[str] = set()
        for media_id in linked_media_ids:
            point = get_point(MEDIA_COLLECTION, media_id)
            if point and point.payload:
                phash = point.payload.get("phash")
                if phash:
                    phashes.add(str(phash))
                else:
                    phashes.add(str(media_id))
            else:
                phashes.add(str(media_id))
        return len(phashes)

    def _confidence_event_count(self, conn, claim_id: str) -> int:
        cutoff = cutoff_days(VOLATILITY_WINDOW_DAYS).isoformat() + "Z"
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM events
            WHERE claim_id = ? AND event_type IN (?, ?)
            AND timestamp >= ?
            """,
            (claim_id, "confidence", "decay", cutoff),
        ).fetchone()
        return safe_int(row["cnt"] if row else 0)
