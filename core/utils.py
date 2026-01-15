import hashlib
import re
from datetime import datetime
from typing import Iterable, List


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    length = 0
    for sentence in sentences:
        if length + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = [sentence]
            length = len(sentence)
        else:
            current.append(sentence)
            length += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return [clean_text(chunk) for chunk in chunks if chunk.strip()]


def uniq_list(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered
