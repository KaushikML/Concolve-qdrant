import io
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from PIL import Image
import streamlit as st
import requests

from agents.orchestrator import run_claim_evolution_agent
from agents.utils import parse_iso
from core.config import settings
from core.utils import clean_text
from ingestion.ingest_meme import ingest_meme
from ingestion.ingest_text import ingest_text
from memory.decay import apply_decay
from models.image_embedder import get_image_embedder
from models.llm_reasoner import generate_deduction
from models.ocr import extract_text
from models.stance_classifier import classify_stance_with_scores
from models.text_embedder import get_text_embedder
from qdrant_store.collections import (
    CLAIMS_COLLECTION,
    EVIDENCE_COLLECTION,
    MEDIA_COLLECTION,
    ensure_collections,
    reset_collections,
)
from qdrant_store.client import get_client
from qdrant_store.crud import search_vectors, scroll_points
from storage.sqlite import get_connection, reset_db


# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="Concolve - Misinformation Correlation",
    layout="wide",
)

ensure_collections()


# --------------------------------------------------
# Cached models
# --------------------------------------------------
@st.cache_resource
def _get_text_embedder():
    return get_text_embedder()


@st.cache_resource
def _get_image_embedder():
    return get_image_embedder()


# --------------------------------------------------
# Corpus status
# --------------------------------------------------
def get_collection_counts() -> Dict[str, int]:
    client = get_client()
    return {
        CLAIMS_COLLECTION: client.count(CLAIMS_COLLECTION).count,
        EVIDENCE_COLLECTION: client.count(EVIDENCE_COLLECTION).count,
        MEDIA_COLLECTION: client.count(MEDIA_COLLECTION).count,
    }


def show_corpus_status():
    counts = get_collection_counts()
    st.caption(
        f"Corpus size — claims: {counts[CLAIMS_COLLECTION]}, "
        f"evidence: {counts[EVIDENCE_COLLECTION]}, "
        f"memes: {counts[MEDIA_COLLECTION]}"
    )


# --------------------------------------------------
# Agent insights helpers
# --------------------------------------------------
def _is_recent(ts: str, hours: int = 24) -> bool:
    dt = parse_iso(ts)
    if not dt:
        return False
    return dt >= datetime.utcnow() - timedelta(hours=hours)


def _load_claim_payloads() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    offset = None
    while True:
        points, next_offset = scroll_points(CLAIMS_COLLECTION, limit=100, offset=offset)
        for point in points:
            payload = point.payload or {}
            payload["claim_id"] = payload.get("canonical_claim_id", point.id)
            rows.append(payload)
        if next_offset is None:
            break
        offset = next_offset
    return rows


def _get_recent_agent_events(limit: int = 50) -> List[Dict[str, object]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT timestamp, claim_id, event_type, delta, reason, source_id, agent_name
        FROM events
        WHERE agent_name IS NOT NULL OR event_type LIKE 'agent_%'
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def _check_ollama_status() -> Dict[str, object]:
    if not settings.use_ollama:
        return {"status": "disabled", "details": "USE_OLLAMA=false"}
    try:
        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        response.raise_for_status()
        payload = response.json()
        models = [model.get("name") for model in payload.get("models", []) if model.get("name")]
        return {
            "status": "ok",
            "details": f"models: {', '.join(models) if models else 'none'}",
        }
    except Exception as exc:
        return {"status": "error", "details": str(exc)}


def _normalize_ollama_word(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if "support" in lowered:
        return "support"
    if "contradict" in lowered:
        return "contradict"
    if "mention" in lowered:
        return "mention"
    return cleaned.split(" ", 1)[0]


def _test_ollama_generate() -> Dict[str, object]:
    if not settings.use_ollama:
        return {"status": "disabled", "details": "USE_OLLAMA=false"}
    try:
        response = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": "Reply with exactly one word: support.",
                "options": {"num_predict": 4, "temperature": 0, "stop": ["\n"]},
                "stream": False,
            },
            timeout=settings.ollama_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("response", "")
        word = _normalize_ollama_word(str(raw))
        if not word:
            word = "No response from Ollama."
        return {"status": "ok", "details": word}
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "details": "Ollama request timed out. Increase OLLAMA_TIMEOUT if needed.",
        }
    except Exception as exc:
        return {"status": "error", "details": str(exc)}


# --------------------------------------------------
# Verdict helpers
# --------------------------------------------------
def _init_verdict():
    return {
        "support_count": 0,
        "contradict_count": 0,
        "mention_count": 0,
        "support_score": 0.0,
        "contradict_score": 0.0,
        "mention_score": 0.0,
    }


def _update_verdict(stats, stance, scores):
    stats[f"{stance}_count"] += 1
    for k in ["support", "contradict", "mention"]:
        stats[f"{k}_score"] += scores.get(k, 0.0)


def _finalize_verdict(stats):
    s, c = stats["support_count"], stats["contradict_count"]

    if s + c < 2:
        label = "Inconclusive"
    elif c > s:
        label = "False (corpus-contradicted)"
    elif s > c:
        label = "True (corpus-supported)"
    else:
        label = "Mixed"

    stats["label"] = label
    return stats


# --------------------------------------------------
# Evidence handling
# --------------------------------------------------
def _push_evidence(ev, query, evidence, verdict, seen):
    payload = ev.payload or {}
    eid = str(payload.get("evidence_id", ev.id))
    if eid in seen:
        return

    seen.add(eid)
    snippet = payload.get("snippet_text", "")
    stance, scores = classify_stance_with_scores(snippet, query)
    _update_verdict(verdict, stance, scores)

    evidence[stance].append(
        {
            "evidence_id": eid,
            "snippet_text": snippet,
            "source_id": payload.get("source_id", ""),
            "score": round(ev.score, 4),
            "stance_score": round(scores.get(stance, 0.0), 4),
        }
    )


def _truncate_text(text: str, limit: int = 160) -> str:
    cleaned = clean_text(text or "")
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


def _meme_hit_rows(hits):
    rows = []
    for hit in hits:
        payload = hit.payload or {}
        rows.append(
            {
                "media_id": payload.get("media_id", str(hit.id)),
                "score": round(hit.score, 4),
                "phash": payload.get("phash", ""),
                "timestamp": payload.get("timestamp", ""),
                "ocr_preview": _truncate_text(payload.get("ocr_text", "")),
                "linked_claim_ids": payload.get("linked_claim_ids", []),
                "source_id": payload.get("source_id", ""),
            }
        )
    return rows


def retrieve_by_claim_text(query: str):
    embedder = _get_text_embedder()
    vector = embedder.embed([query])[0].tolist()

    claim_hits = search_vectors(CLAIMS_COLLECTION, "text_dense", vector, limit=5)

    evidence = {"support": [], "contradict": [], "mention": []}
    verdict = _init_verdict()
    seen = set()

    claim_rows = []
    for hit in claim_hits:
        payload = hit.payload or {}
        cid = payload.get("canonical_claim_id", hit.id)
        claim_rows.append(
            {
                "claim_id": cid,
                "claim_text": payload.get("claim_text", ""),
                "score": round(hit.score, 4),
            }
        )

        filters = {"must": [{"key": "claim_id", "match": {"value": cid}}]}
        ev_hits = search_vectors(
            EVIDENCE_COLLECTION, "snippet_dense", vector, limit=20, filters=filters
        )

        for ev in ev_hits:
            _push_evidence(ev, query, evidence, verdict, seen)

    return claim_rows, evidence, _finalize_verdict(verdict)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Concolve: Multimodal Misinformation Correlation Engine")

page = st.sidebar.radio(
    "Navigation", ["Analyze Claim/Text", "Analyze Meme", "Ingest Corpus", "Agent Insights"]
)

# --------------------------------------------------
# Analyze Claim/Text (FIXED)
# --------------------------------------------------
if page == "Analyze Claim/Text":
    show_corpus_status()

    query = st.text_area(
        "Enter a claim or text",
        placeholder="Deep learning is not a part of machine learning",
    )

    analyze = st.button("🔍 Analyze Claim")

    if analyze:
        if not query.strip():
            st.warning("Please enter a claim.")
        else:
            with st.spinner("Analyzing against corpus..."):
                claims, evidence, verdict = retrieve_by_claim_text(query)

            st.subheader("Verdict")
            st.markdown(f"### 🧠 {verdict['label']}")

            st.subheader("Matched Claims")
            if claims:
                st.dataframe(pd.DataFrame(claims))
            else:
                st.info("No similar claims found.")

            st.subheader("LLM Deduction (Ollama)")
            if settings.use_ollama:
                with st.spinner("Generating deduction with Ollama..."):
                    deduction = generate_deduction(query, claims, evidence)
                if deduction["status"] == "ok":
                    st.markdown(deduction["text"])
                elif deduction["status"] == "disabled":
                    st.caption(deduction["text"])
                else:
                    st.warning(f"Ollama error — {deduction['text']}")
            else:
                st.caption("Ollama disabled. Enable USE_OLLAMA=true to show deduction.")

            st.subheader("Evidence")
            for stance, rows in evidence.items():
                st.markdown(f"**{stance.title()}**")
                if rows:
                    st.dataframe(pd.DataFrame(rows))
                else:
                    st.caption("No evidence found.")


# --------------------------------------------------
# Analyze Meme
# --------------------------------------------------
elif page == "Analyze Meme":
    show_corpus_status()
    upload = st.file_uploader("Upload Meme Image", ["png", "jpg", "jpeg"])

    if upload:
        image = Image.open(io.BytesIO(upload.read())).convert("RGB")
        st.image(image, width="stretch")
        analyze = st.button("🔍 Analyze Meme")

        if analyze:
            with st.spinner("Extracting text and searching for matches..."):
                ocr_text = clean_text(extract_text(image))
                image_vector = _get_image_embedder().embed([image])[0].tolist()
                image_hits = search_vectors(
                    MEDIA_COLLECTION, "image_dense", image_vector, limit=5
                )
                text_hits = []
                if ocr_text:
                    text_vector = _get_text_embedder().embed([ocr_text])[0].tolist()
                    text_hits = search_vectors(
                        MEDIA_COLLECTION, "ocr_text_dense", text_vector, limit=5
                    )

            st.subheader("OCR Text")
            if ocr_text:
                st.write(ocr_text)
            else:
                st.caption("No text detected in this meme.")

            st.subheader("Similar Memes (Image)")
            if image_hits:
                st.dataframe(pd.DataFrame(_meme_hit_rows(image_hits)))
            else:
                st.info("No similar memes found using image similarity.")

            st.subheader("Similar Memes (OCR Text)")
            if ocr_text:
                if text_hits:
                    st.dataframe(pd.DataFrame(_meme_hit_rows(text_hits)))
                else:
                    st.info("No similar memes found using OCR text similarity.")
            else:
                st.caption("OCR text is empty; skipping OCR similarity search.")

            if ocr_text:
                with st.spinner("Matching OCR text against claims..."):
                    claims, evidence, verdict = retrieve_by_claim_text(ocr_text)

                st.subheader("Verdict (OCR Text)")
                st.markdown(f"### 🧠 {verdict['label']}")

                st.subheader("Matched Claims (OCR Text)")
                if claims:
                    st.dataframe(pd.DataFrame(claims))
                else:
                    st.info("No similar claims found from OCR text.")

                st.subheader("Evidence (OCR Text)")
                for stance, rows in evidence.items():
                    st.markdown(f"**{stance.title()}**")
                    if rows:
                        st.dataframe(pd.DataFrame(rows))
                    else:
                        st.caption("No evidence found.")
            else:
                st.caption("OCR text is empty; skipping claim matching.")


# --------------------------------------------------
# Agent Insights
# --------------------------------------------------
elif page == "Agent Insights":
    show_corpus_status()
    st.header("Agent Insights")

    st.subheader("Ollama Status")
    status = _check_ollama_status()
    if status["status"] == "ok":
        st.success(f"Ollama OK — {status['details']}")
    elif status["status"] == "disabled":
        st.info(f"Ollama disabled — {status['details']}")
    else:
        st.warning(f"Ollama error — {status['details']}")

    if st.button("Test Ollama Generate"):
        with st.spinner("Testing Ollama..."):
            test_status = _test_ollama_generate()
        if test_status["status"] == "ok":
            st.success(f"Response: {test_status['details']}")
        elif test_status["status"] == "disabled":
            st.info(f"Ollama disabled — {test_status['details']}")
        else:
            st.warning(f"Ollama error — {test_status['details']}")

    if st.button("Run Agent Now"):
        with st.spinner("Running claim evolution agent..."):
            summary = run_claim_evolution_agent(force_full_scan=False)
        st.success(
            f"Agent run complete — updated {summary.get('claims_updated', 0)} claims, "
            f"high alerts {summary.get('high_alerts', 0)}"
        )

    claims = _load_claim_payloads()
    if not claims:
        st.info("No claims available yet. Ingest content to see agent insights.")
    else:
        updated_recent = sum(
            1 for claim in claims if _is_recent(str(claim.get("last_agent_update_ts", "")))
        )
        disputed = sum(1 for claim in claims if claim.get("status") == "disputed")
        high_alerts = sum(1 for claim in claims if claim.get("alert_level") == "high")

        st.subheader("Agent Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Claims Updated (24h)", updated_recent)
        col2.metric("Disputed Claims", disputed)
        col3.metric("High Alerts", high_alerts)

        st.subheader("Trending Claims")
        trend_rows = sorted(
            claims, key=lambda c: float(c.get("trend_score", 0.0)), reverse=True
        )[:10]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "claim_id": row.get("claim_id"),
                        "claim_text": row.get("claim_text"),
                        "trend_score": row.get("trend_score", 0.0),
                        "contradiction_ratio": row.get("contradiction_ratio", 0.0),
                        "alert_level": row.get("alert_level", "low"),
                    }
                    for row in trend_rows
                ]
            )
        )

        st.subheader("Disputed Claims")
        dispute_rows = sorted(
            claims, key=lambda c: float(c.get("contradiction_ratio", 0.0)), reverse=True
        )[:10]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "claim_id": row.get("claim_id"),
                        "claim_text": row.get("claim_text"),
                        "contradiction_ratio": row.get("contradiction_ratio", 0.0),
                        "alert_level": row.get("alert_level", "low"),
                    }
                    for row in dispute_rows
                ]
            )
        )

        st.subheader("Recent Agent Events")
        events = _get_recent_agent_events()
        if events:
            st.dataframe(pd.DataFrame(events))
        else:
            st.caption("No agent events logged yet.")

        st.subheader("Explainability Panel")
        option_map = {}
        for row in claims:
            claim_id = str(row.get("claim_id"))
            label = f"{row.get('claim_text', '')[:60]}... ({claim_id})"
            option_map[label] = row
        selected = st.selectbox("Select a claim", list(option_map.keys()))
        selected_claim = option_map.get(selected)
        if selected_claim:
            st.json(
                {
                    "claim_text": selected_claim.get("claim_text"),
                    "trend_score": selected_claim.get("trend_score"),
                    "contradiction_ratio": selected_claim.get("contradiction_ratio"),
                    "support_count": selected_claim.get("support_count"),
                    "contradict_count": selected_claim.get("contradict_count"),
                    "meme_variant_count": selected_claim.get("meme_variant_count"),
                    "volatility_score": selected_claim.get("volatility_score"),
                    "alert_level": selected_claim.get("alert_level"),
                    "status": selected_claim.get("status"),
                    "last_agent_update_ts": selected_claim.get("last_agent_update_ts"),
                }
            )


# --------------------------------------------------
# Ingest Corpus
# --------------------------------------------------
elif page == "Ingest Corpus":
    st.header("Ingest Corpus")

    memes = st.file_uploader("Upload Memes", ["png", "jpg"], accept_multiple_files=True)
    texts = st.file_uploader("Upload Text Files", ["txt"], accept_multiple_files=True)

    if st.button("Ingest"):
        temp_dir = tempfile.gettempdir()
        for f in texts or []:
            path = os.path.join(temp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            ingest_text(path)

        for f in memes or []:
            path = os.path.join(temp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            ingest_meme(path)

        st.success("Ingestion completed")

    if st.button("Run decay"):
        updated = apply_decay()
        st.info(f"Decay applied to {updated} claims")

    st.divider()
    st.subheader("Danger zone")
    st.caption("Remove all ingested corpus data from Qdrant and the local metadata store.")
    confirm_clear = st.checkbox("I understand this will permanently delete all ingested data.")
    if st.button("Remove all corpus data", type="primary", disabled=not confirm_clear):
        with st.spinner("Clearing corpus..."):
            reset_collections()
            reset_db()
        st.success("Corpus cleared.")
