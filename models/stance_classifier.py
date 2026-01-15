from typing import Literal
import re
import requests

from core.config import settings


Stance = Literal["support", "contradict", "mention"]


def _rule_based_stance(snippet: str, claim: str) -> Stance:
    snippet_lower = snippet.lower()
    if any(term in snippet_lower for term in ["debunk", "false", "incorrect", "misleading", "no evidence"]):
        return "contradict"
    if any(term in snippet_lower for term in ["confirmed", "true", "verified", "evidence shows", "supports"]):
        return "support"
    return "mention"


def _ollama_stance(snippet: str, claim: str) -> Stance:
    prompt = (
        "Classify stance of snippet toward claim as support, contradict, or mention. "
        "Respond with only one word.\n\n"
        f"Claim: {claim}\nSnippet: {snippet}"
    )
    response = requests.post(
        f"{settings.ollama_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    text = data.get("response", "mention").strip().lower()
    if text in {"support", "contradict", "mention"}:
        return text  # type: ignore[return-value]
    if re.search(r"contradict|refute|deny", text):
        return "contradict"
    if re.search(r"support|confirm|verify", text):
        return "support"
    return "mention"


def classify_stance(snippet: str, claim: str) -> Stance:
    if settings.use_ollama:
        try:
            return _ollama_stance(snippet, claim)
        except Exception:
            return _rule_based_stance(snippet, claim)
    return _rule_based_stance(snippet, claim)
