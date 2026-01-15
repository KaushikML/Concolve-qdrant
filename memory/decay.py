from datetime import datetime, timedelta

from core.config import settings
from qdrant_store.collections import CLAIMS_COLLECTION
from qdrant_store.crud import scroll_points, update_payload
from memory.events import log_event


def apply_decay() -> int:
    decay_before = datetime.utcnow() - timedelta(days=settings.decay_days)
    decay_before_iso = decay_before.isoformat() + "Z"
    updated = 0
    offset = None
    while True:
        points, next_offset = scroll_points(CLAIMS_COLLECTION, limit=50, offset=offset)
        for point in points:
            last_seen = point.payload.get("last_seen_ts", "")
            if last_seen and last_seen < decay_before_iso:
                current = float(point.payload.get("confidence", 0.5))
                new_conf = current + (0.5 - current) * 0.1
                update_payload(
                    CLAIMS_COLLECTION,
                    point.id,
                    {"confidence": new_conf},
                )
                log_event(str(point.id), "decay", new_conf - current, "decay toward neutral")
                updated += 1
        if next_offset is None:
            break
        offset = next_offset
    return updated
