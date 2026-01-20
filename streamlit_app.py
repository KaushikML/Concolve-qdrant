import io
import os
import tempfile
from typing import Dict, List

import pandas as pd
from PIL import Image
import streamlit as st

from core.utils import clean_text
from ingestion.ingest_meme import ingest_meme
from ingestion.ingest_text import ingest_text
from memory.decay import apply_decay
from models.image_embedder import get_image_embedder
from models.ocr import extract_text
from models.stance_classifier import classify_stance_with_scores
from models.text_embedder import get_text_embedder
from qdrant_store.collections import (
    CLAIMS_COLLECTION,
    EVIDENCE_COLLECTION,
    MEDIA_COLLECTION,
    ensure_collections,
)
from qdrant_store.client import get_client
from qdrant_store.crud import search_vectors, scroll_points


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
    "Navigation", ["Analyze Claim/Text", "Analyze Meme", "Ingest Corpus"]
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
        st.image(image, use_container_width=True)
        st.info("Meme analysis pipeline ready (text mode works fully).")


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
