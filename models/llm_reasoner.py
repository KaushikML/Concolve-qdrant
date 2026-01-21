from typing import Any, Dict, List
import json

import requests

from core.config import settings


def _clean_snippet(text: str, max_chars: int = 280) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars]


def _format_claims(claim_rows: List[Dict[str, Any]], max_claims: int = 5) -> str:
    if not claim_rows:
        return "- none"
    lines = []
    for row in claim_rows[:max_claims]:
        claim_text = _clean_snippet(str(row.get("claim_text", "")))
        score = row.get("score", 0.0)
        lines.append(f"- {claim_text} (score={score})")
    return "\n".join(lines)


def _format_evidence(evidence: Dict[str, List[Dict[str, Any]]], max_per_stance: int = 3) -> str:
    sections = []
    for stance in ("support", "contradict", "mention"):
        rows = sorted(
            evidence.get(stance, []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )[:max_per_stance]
        sections.append(f"{stance.title()}:")
        if not rows:
            sections.append("- none")
            continue
        for row in rows:
            snippet = _clean_snippet(str(row.get("snippet_text", "")))
            source_id = row.get("source_id", "")
            sections.append(f"- {snippet} (source={source_id})")
    return "\n".join(sections)


def build_deduction_prompt(
    query: str,
    claim_rows: List[Dict[str, Any]],
    evidence: Dict[str, List[Dict[str, Any]]],
) -> str:
    claims_block = _format_claims(claim_rows)
    evidence_block = _format_evidence(evidence)
    return (
        "You are a cautious fact-checking assistant. Use only the provided evidence.\n"
        "Return a short deduction with one label: Supported, Contradicted, Mixed, or Inconclusive.\n"
        "Then provide 2-4 sentences explaining why. If evidence is weak or conflicting, say so.\n\n"
        f"Claim: {query}\n\n"
        f"Matched claims:\n{claims_block}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Answer format:\n"
        "Label: <Supported|Contradicted|Mixed|Inconclusive>\n"
        "Reason: <short explanation>\n"
    )


def generate_deduction(
    query: str,
    claim_rows: List[Dict[str, Any]],
    evidence: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, str]:
    if not settings.use_ollama:
        return {"status": "disabled", "text": "Ollama disabled (USE_OLLAMA=false)."}
    prompt = build_deduction_prompt(query, claim_rows, evidence)
    try:
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": settings.ollama_stream,
            "options": {
                "temperature": settings.ollama_temperature,
                "num_predict": settings.ollama_num_predict,
            },
        }
        response = requests.post(
            f"{settings.ollama_url}/api/generate",
            json=payload,
            stream=settings.ollama_stream,
            timeout=(5, settings.ollama_timeout),
        )
        response.raise_for_status()
        if settings.ollama_stream:
            chunks = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("response", "")
                if chunk:
                    chunks.append(chunk)
                if data.get("done"):
                    break
            text = "".join(chunks).strip()
        else:
            data = response.json()
            text = str(data.get("response", "")).strip()
        if not text:
            text = "No response from Ollama."
        return {"status": "ok", "text": text}
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "text": (
                "Ollama request timed out. Increase OLLAMA_TIMEOUT to allow longer"
                " generations."
            ),
        }
    except Exception as exc:
        return {"status": "error", "text": str(exc)}
