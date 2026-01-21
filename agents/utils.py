from datetime import datetime, timedelta
from typing import Iterable, List, Optional


TREND_WINDOW_DAYS = 7
CONTRADICTION_WINDOW_DAYS = 30
VOLATILITY_WINDOW_DAYS = 30
CONTRADICTION_THRESHOLD = 0.6
VOLATILITY_EVENT_CAP = 5
VOLATILITY_ALERT_THRESHOLD = 0.7
TREND_ALERT_THRESHOLD = 6


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    normalized = ts[:-1] if ts.endswith("Z") else ts
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def cutoff_days(days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=days)


def is_within_days(ts: Optional[str], days: int) -> bool:
    dt = parse_iso(ts)
    if not dt:
        return False
    return dt >= cutoff_days(days)


def chunk_list(items: Iterable[str], size: int = 900) -> List[List[str]]:
    chunked: List[List[str]] = []
    buffer: List[str] = []
    for item in items:
        buffer.append(item)
        if len(buffer) >= size:
            chunked.append(buffer)
            buffer = []
    if buffer:
        chunked.append(buffer)
    return chunked


def safe_float(value: Optional[object], default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def safe_int(value: Optional[object], default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def contradiction_ratio(support: int, contradict: int, eps: float = 1e-6) -> float:
    return float(contradict) / float(support + contradict + eps)


def volatility_score(event_count: int) -> float:
    if event_count <= 0:
        return 0.0
    return min(1.0, float(event_count) / float(VOLATILITY_EVENT_CAP))


def compute_alert_level(trend: float, contradiction: float, volatility: float) -> str:
    if contradiction >= CONTRADICTION_THRESHOLD and trend >= 3:
        return "high"
    if contradiction >= CONTRADICTION_THRESHOLD or trend >= TREND_ALERT_THRESHOLD:
        return "medium"
    if volatility >= VOLATILITY_ALERT_THRESHOLD and trend >= 2:
        return "medium"
    return "low"
