import uuid
from typing import Dict, List, Tuple

from core.config import settings
from core.utils import now_iso, uniq_list
from memory.events import log_event
from qdrant_store.collections import CLAIMS_COLLECTION
from qdrant_store.crud import search_vectors, upsert_point, update_payload


def canonicalize_claim(
    claim_text: str,
    embedding: List[float],
    source_type: str,
) -> Tuple[str, bool]:
    matches = search_vectors(CLAIMS_COLLECTION, "text_dense", embedding, limit=5)
    if matches and matches[0].score >= settings.claim_sim_threshold:
        point = matches[0]
        payload = point.payload
        mention_count = int(payload.get("mention_count", 1)) + 1
        source_types = uniq_list(payload.get("source_types", []) + [source_type])
        update_payload(
            CLAIMS_COLLECTION,
            point.id,
            {
                "mention_count": mention_count,
                "last_seen_ts": now_iso(),
                "source_types": source_types,
            },
        )
        log_event(str(point.id), "merge", 0.0, "claim merged", None)
        return str(point.id), True

    claim_id = str(uuid.uuid4())
    payload = {
        "canonical_claim_id": claim_id,
        "claim_text": claim_text,
        "first_seen_ts": now_iso(),
        "last_seen_ts": now_iso(),
        "mention_count": 1,
        "source_types": [source_type],
        "support_count": 0,
        "contradict_count": 0,
        "confidence": 0.5,
        "status": "unverified",
        "linked_evidence_ids": [],
        "linked_media_ids": [],
        "trend_score": 0.0,
        "contradiction_ratio": 0.0,
        "meme_variant_count": 0,
        "volatility_score": 0.0,
        "alert_level": "low",
        "last_agent_update_ts": None,
    }
    upsert_point(CLAIMS_COLLECTION, claim_id, {"text_dense": embedding}, payload)
    log_event(claim_id, "create", 0.0, "new canonical claim", None)
    return claim_id, False
