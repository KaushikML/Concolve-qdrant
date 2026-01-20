from typing import List
import re
import requests

from core.config import settings
from core.utils import clean_text


def _rule_based_extract(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims = []
    for sentence in sentences:
        sentence = clean_text(sentence)
        if len(sentence) < 15:
            continue
        if any(keyword in sentence.lower() for keyword in ["claims", "says", "reports", "rumor", "hoax", "fake"]):
            claims.append(sentence)
        elif sentence.endswith(".") or sentence.endswith("!") or sentence.endswith("?"):
            claims.append(sentence)
    return claims[:5]


def _ollama_extract(text: str) -> List[str]:
    prompt = (
        "Extract up to 5 concise claim statements from the text. "
        "Return as a JSON list of strings.\n\nText:\n"
        f"{text}"
    )
    response = requests.post(
        f"{settings.ollama_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("response", "[]")
    matches = re.findall(r"\[(.*)\]", content, re.DOTALL)
    if not matches:
        return _rule_based_extract(text)
    items = re.findall(r"\"(.*?)\"", matches[0])
    return [clean_text(item) for item in items if item]


def extract_claims(text: str) -> List[str]:
    if settings.use_ollama:
        try:
            return _ollama_extract(text)
        except Exception:
            return _rule_based_extract(text)
    return _rule_based_extract(text)
