from functools import lru_cache
from typing import Dict, Literal, Tuple
import os
import re

import requests
import torch
from transformers import pipeline

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


@lru_cache(maxsize=1)
def _get_nli_pipeline():
    model_name = os.getenv("NLI_MODEL_NAME", "facebook/bart-large-mnli")
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model_name, device=device)


def _normalize_nli_scores(results) -> Dict[str, float]:
    scores = {"support": 0.0, "contradict": 0.0, "mention": 0.0}
    for item in results:
        label = str(item.get("label", "")).lower()
        if "entail" in label:
            scores["support"] = float(item.get("score", 0.0))
        elif "contrad" in label:
            scores["contradict"] = float(item.get("score", 0.0))
        elif "neutral" in label:
            scores["mention"] = float(item.get("score", 0.0))
    if all(value == 0.0 for value in scores.values()):
        scores["mention"] = 1.0
    return scores


def _nli_stance(snippet: str, claim: str) -> Tuple[Stance, Dict[str, float]]:
    classifier = _get_nli_pipeline()
    try:
        results = classifier(
            {"text": snippet, "text_pair": claim},
            top_k=None,
            truncation=True,
        )
    except TypeError:
        results = classifier(
            (snippet, claim),
            top_k=None,
            truncation=True,
        )
    if isinstance(results, list) and results:
        results = results[0]
    scores = _normalize_nli_scores(results or [])
    stance = max(scores, key=scores.get)
    return stance, scores


def classify_stance_with_scores(snippet: str, claim: str) -> Tuple[Stance, Dict[str, float]]:
    if not snippet.strip() or not claim.strip():
        return "mention", {"support": 0.0, "contradict": 0.0, "mention": 1.0}
    if settings.use_ollama:
        try:
            stance = _ollama_stance(snippet, claim)
            scores = {"support": 0.0, "contradict": 0.0, "mention": 0.0}
            scores[stance] = 1.0
            return stance, scores
        except Exception:
            stance = _rule_based_stance(snippet, claim)
            scores = {"support": 0.0, "contradict": 0.0, "mention": 0.0}
            scores[stance] = 1.0
            return stance, scores
    try:
        return _nli_stance(snippet, claim)
    except Exception:
        stance = _rule_based_stance(snippet, claim)
        scores = {"support": 0.0, "contradict": 0.0, "mention": 0.0}
        scores[stance] = 1.0
        return stance, scores


def classify_stance(snippet: str, claim: str) -> Stance:
    stance, _ = classify_stance_with_scores(snippet, claim)
    return stance
