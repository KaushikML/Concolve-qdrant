import uuid
from typing import Dict, List

from PIL import Image

from core.utils import clean_text, now_iso, uniq_list
from ingestion.dedup import meme_phash
from memory.canonicalize import canonicalize_claim
from memory.events import log_event
from models.claim_extractor import extract_claims
from models.image_embedder import get_image_embedder
from models.ocr import extract_text
from models.text_embedder import get_text_embedder
from qdrant_store.collections import MEDIA_COLLECTION
from qdrant_store.crud import get_point, search_vectors, upsert_point, update_payload
from storage.sqlite import get_connection
from agents.orchestrator import run_claim_evolution_agent


def ingest_meme(path: str, source_type: str = "meme") -> Dict[str, int]:
    image = Image.open(path).convert("RGB")
    phash = meme_phash(image)
    text_embedder = get_text_embedder()
    image_embedder = get_image_embedder()
    conn = get_connection()

    image_vector = image_embedder.embed([image])[0].tolist()
    ocr_text = clean_text(extract_text(image))
    ocr_vector = text_embedder.embed([ocr_text or "no text"])[0].tolist()

    duplicates = search_vectors(
        MEDIA_COLLECTION,
        "image_dense",
        image_vector,
        limit=3,
    )
    for dup in duplicates:
        if dup.payload and dup.payload.get("phash") == phash:
            return {"memes_ingested": 0, "memes_deduped": 1}

    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO sources (source_id, source_type, title, timestamp, url, text_hash) VALUES (?, ?, ?, ?, ?, ?)",
            (path, source_type, path, now_iso(), None, phash),
        )

    claims = extract_claims(ocr_text or "")
    linked_claim_ids: List[str] = []
    for claim in claims:
        emb = text_embedder.embed([claim])[0].tolist()
        claim_id, merged = canonicalize_claim(claim, emb, source_type)
        linked_claim_ids.append(claim_id)
        with conn:
            conn.execute(
                "INSERT INTO claim_links (source_id, claim_id) VALUES (?, ?)",
                (path, claim_id),
            )
        if merged:
            log_event(claim_id, "reinforce", 0.0, "meme mention", path)

    media_id = str(uuid.uuid4())
    payload = {
        "media_id": media_id,
        "source_id": path,
        "timestamp": now_iso(),
        "phash": phash,
        "ocr_text": ocr_text,
        "linked_claim_ids": linked_claim_ids,
    }
    upsert_point(
        MEDIA_COLLECTION,
        media_id,
        {"image_dense": image_vector, "ocr_text_dense": ocr_vector},
        payload,
    )

    for claim_id in linked_claim_ids:
        existing = get_point("claims", claim_id)
        current = existing.payload.get("linked_media_ids", []) if existing else []
        update_payload(
            "claims",
            claim_id,
            {"linked_media_ids": uniq_list(current + [media_id])},
        )

    run_claim_evolution_agent([path], force_full_scan=False)
    return {"memes_ingested": 1, "memes_deduped": 0}
