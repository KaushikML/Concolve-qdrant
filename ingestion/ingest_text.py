import uuid
from typing import Dict, List

from core.utils import chunk_text, now_iso
from ingestion.dedup import text_hash
from memory.canonicalize import canonicalize_claim
from memory.confidence import update_confidence
from memory.events import log_event
from models.claim_extractor import extract_claims
from models.stance_classifier import classify_stance
from models.text_embedder import get_text_embedder
from qdrant_store.collections import EVIDENCE_COLLECTION
from qdrant_store.crud import get_point, upsert_point, update_payload
from storage.sqlite import get_connection


def ingest_text(path: str, source_type: str = "article") -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text_embedder = get_text_embedder()
    conn = get_connection()
    source_id = path
    text_digest = text_hash(text)

    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (source_id, source_type, title, timestamp, url, text_hash) VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, source_type, path, now_iso(), None, text_digest),
        )

    chunks = chunk_text(text)
    claim_candidates = extract_claims(text)
    claim_ids: List[str] = []
    for claim in claim_candidates:
        emb = text_embedder.embed([claim])[0].tolist()
        claim_id, merged = canonicalize_claim(claim, emb, source_type)
        claim_ids.append(claim_id)
        with conn:
            conn.execute(
                "INSERT INTO claim_links (source_id, claim_id) VALUES (?, ?)",
                (source_id, claim_id),
            )
        if merged:
            log_event(claim_id, "reinforce", 0.0, "text mention", source_id)

    evidence_added = 0
    for chunk in chunks:
        for claim_id in claim_ids:
            claim_point = get_point("claims", claim_id)
            claim_text = claim_point.payload.get("claim_text", "") if claim_point else ""
            stance = classify_stance(chunk, claim_text)
            evidence_id = str(uuid.uuid4())
            snippet_vector = text_embedder.embed([chunk])[0].tolist()
            payload = {
                "evidence_id": evidence_id,
                "claim_id": claim_id,
                "snippet_text": chunk,
                "stance": stance,
                "source_id": source_id,
                "source_type": source_type,
                "timestamp": now_iso(),
                "url": None,
                "credibility_tier": "C",
            }
            upsert_point(
                EVIDENCE_COLLECTION,
                evidence_id,
                {"snippet_dense": snippet_vector},
                payload,
            )
            evidence_added += 1
            if claim_point:
                current_conf = float(claim_point.payload.get("confidence", 0.5))
                new_conf, delta = update_confidence(current_conf, stance, "C")
                support_count = int(claim_point.payload.get("support_count", 0)) + (
                    1 if stance == "support" else 0
                )
                contradict_count = int(claim_point.payload.get("contradict_count", 0)) + (
                    1 if stance == "contradict" else 0
                )
                update_payload(
                    "claims",
                    claim_id,
                    {
                        "confidence": new_conf,
                        "support_count": support_count,
                        "contradict_count": contradict_count,
                    },
                )
                log_event(claim_id, "confidence", delta, f"stance {stance}", source_id)

    return {"evidence_added": evidence_added, "claims_created": len(set(claim_ids))}
