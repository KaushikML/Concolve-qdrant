import io
from typing import Dict, List

import pandas as pd
from PIL import Image
import streamlit as st

from core.utils import clean_text
from ingestion.ingest_meme import ingest_meme
from ingestion.ingest_text import ingest_text
from memory.decay import apply_decay
from models.claim_extractor import extract_claims
from models.image_embedder import get_image_embedder
from models.ocr import extract_text
from models.text_embedder import get_text_embedder
from qdrant_store.collections import CLAIMS_COLLECTION, EVIDENCE_COLLECTION, MEDIA_COLLECTION, ensure_collections
from qdrant_store.crud import get_point, search_vectors
from qdrant_store.client import get_client


st.set_page_config(page_title="Concolve - Misinformation Correlation", layout="wide")

ensure_collections()


@st.cache_resource
def _get_text_embedder():
    return get_text_embedder()


@st.cache_resource
def _get_image_embedder():
    return get_image_embedder()


def build_trace(points, path: str, collection: str) -> List[Dict[str, str]]:
    trace = []
    for point in points:
        payload_preview = {}
        if point.payload:
            payload_preview = {
                key: point.payload.get(key)
                for key in ["claim_text", "ocr_text", "snippet_text", "source_id"]
                if key in point.payload
            }
        trace.append(
            {
                "collection": collection,
                "point_id": str(point.id),
                "score": round(point.score, 4),
                "path": path,
                "filters": "",
                "payload_preview": str(payload_preview),
            }
        )
    return trace


def retrieve_by_claim_text(query: str):
    text_embedder = _get_text_embedder()
    query_vector = text_embedder.embed([query])[0].tolist()
    claim_hits = search_vectors(CLAIMS_COLLECTION, "text_dense", query_vector, limit=5)
    evidence = {"support": [], "contradict": [], "mention": []}
    trace = build_trace(claim_hits, "text query", CLAIMS_COLLECTION)

    claim_rows = []
    similar_memes = []
    for hit in claim_hits:
        payload = hit.payload or {}
        claim_id = payload.get("canonical_claim_id", hit.id)
        claim_rows.append(
            {
                "claim_id": claim_id,
                "claim_text": payload.get("claim_text", ""),
                "score": hit.score,
                "last_seen_ts": payload.get("last_seen_ts", ""),
                "confidence": payload.get("confidence", 0.5),
            }
        )
        for media_id in payload.get("linked_media_ids", []):
            media_point = get_point(MEDIA_COLLECTION, media_id)
            if media_point:
                similar_memes.append(
                    {
                        "media_id": media_id,
                        "source_id": media_point.payload.get("source_id", ""),
                        "score": "linked",
                    }
                )
        filters = {
            "must": [
                {"key": "claim_id", "match": {"value": claim_id}},
            ]
        }
        evidence_hits = get_client().search(
            collection_name=EVIDENCE_COLLECTION,
            query_vector=("snippet_dense", query_vector),
            limit=10,
            query_filter=filters,
            with_payload=True,
        )
        trace.extend(build_trace(evidence_hits, "evidence expand", EVIDENCE_COLLECTION))
        for ev in evidence_hits:
            ev_payload = ev.payload or {}
            stance = ev_payload.get("stance", "mention")
            evidence[stance].append(
                {
                    "evidence_id": ev_payload.get("evidence_id", ev.id),
                    "snippet_text": ev_payload.get("snippet_text", ""),
                    "source_id": ev_payload.get("source_id", ""),
                    "score": ev.score,
                }
            )

    return claim_rows, evidence, similar_memes, trace


def retrieve_by_meme(image: Image.Image):
    text_embedder = _get_text_embedder()
    image_embedder = _get_image_embedder()
    ocr_text = clean_text(extract_text(image))
    ocr_vector = text_embedder.embed([ocr_text or "no text"])[0].tolist()
    image_vector = image_embedder.embed([image])[0].tolist()

    meme_hits = search_vectors(MEDIA_COLLECTION, "image_dense", image_vector, limit=6)
    claim_hits = search_vectors(CLAIMS_COLLECTION, "text_dense", ocr_vector, limit=5)

    trace = []
    trace.extend(build_trace(meme_hits, "image similarity", MEDIA_COLLECTION))
    trace.extend(build_trace(claim_hits, "ocr text", CLAIMS_COLLECTION))

    claim_rows = []
    candidate_claim_ids = set()
    for hit in claim_hits:
        payload = hit.payload or {}
        claim_id = payload.get("canonical_claim_id", hit.id)
        candidate_claim_ids.add(claim_id)
        claim_rows.append(
            {
                "claim_id": claim_id,
                "claim_text": payload.get("claim_text", ""),
                "score": hit.score,
                "last_seen_ts": payload.get("last_seen_ts", ""),
                "confidence": payload.get("confidence", 0.5),
            }
        )

    similar_memes = []
    for hit in meme_hits:
        payload = hit.payload or {}
        similar_memes.append(
            {
                "media_id": payload.get("media_id", hit.id),
                "source_id": payload.get("source_id", ""),
                "score": hit.score,
            }
        )
        for claim_id in payload.get("linked_claim_ids", []):
            candidate_claim_ids.add(claim_id)

    evidence = {"support": [], "contradict": [], "mention": []}
    for claim_id in candidate_claim_ids:
        filters = {
            "must": [
                {"key": "claim_id", "match": {"value": claim_id}},
            ]
        }
        evidence_hits = get_client().search(
            collection_name=EVIDENCE_COLLECTION,
            query_vector=("snippet_dense", ocr_vector),
            limit=10,
            query_filter=filters,
            with_payload=True,
        )
        trace.extend(build_trace(evidence_hits, "evidence expand", EVIDENCE_COLLECTION))
        for ev in evidence_hits:
            ev_payload = ev.payload or {}
            stance = ev_payload.get("stance", "mention")
            evidence[stance].append(
                {
                    "evidence_id": ev_payload.get("evidence_id", ev.id),
                    "snippet_text": ev_payload.get("snippet_text", ""),
                    "source_id": ev_payload.get("source_id", ""),
                    "score": ev.score,
                }
            )

    return ocr_text, claim_rows, evidence, similar_memes, trace


st.title("Concolve: Multimodal Misinformation Correlation Engine")

page = st.sidebar.radio("Navigation", ["Analyze Meme", "Analyze Claim/Text", "Ingest Corpus"])

if page == "Analyze Meme":
    st.header("Analyze Meme")
    upload = st.file_uploader("Upload Meme Image", type=["png", "jpg", "jpeg"])
    if upload:
        image = Image.open(io.BytesIO(upload.read())).convert("RGB")
        st.image(image, caption="Uploaded Meme", use_column_width=True)
        ocr_text = clean_text(extract_text(image))
        st.subheader("OCR Text")
        st.code(ocr_text or "(No text detected)")
        st.subheader("Extracted Claim Candidates")
        st.write(extract_claims(ocr_text or ""))

        ocr_text, claim_rows, evidence, similar_memes, trace = retrieve_by_meme(image)
        st.subheader("Matched Canonical Claims")
        st.dataframe(pd.DataFrame(claim_rows))

        st.subheader("Similar Meme Variants")
        if similar_memes:
            st.dataframe(pd.DataFrame(similar_memes))
        else:
            st.info("No similar meme variants found.")

        st.subheader("Evidence Snippets")
        for stance, rows in evidence.items():
            st.markdown(f"**{stance.title()}**")
            st.dataframe(pd.DataFrame(rows))

        st.subheader("Timeline")
        timeline = [
            {
                "claim_id": row["claim_id"],
                "last_seen_ts": row["last_seen_ts"],
            }
            for row in claim_rows
        ]
        st.dataframe(pd.DataFrame(timeline))

        with st.expander("Why this result?"):
            st.dataframe(pd.DataFrame(trace))

elif page == "Analyze Claim/Text":
    st.header("Analyze Claim/Text")
    query = st.text_area("Enter a claim or text")
    if query:
        claim_rows, evidence, similar_memes, trace = retrieve_by_claim_text(query)
        st.subheader("Matched Canonical Claims")
        st.dataframe(pd.DataFrame(claim_rows))

        st.subheader("Evidence Snippets")
        for stance, rows in evidence.items():
            st.markdown(f"**{stance.title()}**")
            st.dataframe(pd.DataFrame(rows))

        st.subheader("Linked Meme Variants")
        if similar_memes:
            st.dataframe(pd.DataFrame(similar_memes))
        else:
            st.info("No linked meme variants found.")

        with st.expander("Why this result?"):
            st.dataframe(pd.DataFrame(trace))

elif page == "Ingest Corpus":
    st.header("Ingest Corpus")
    memes = st.file_uploader("Upload Memes", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    texts = st.file_uploader("Upload Text Files", type=["txt"], accept_multiple_files=True)
    if st.button("Ingest"):
        summary = {
            "memes_ingested": 0,
            "memes_deduped": 0,
            "evidence_added": 0,
            "claims_created": 0,
        }
        for meme in memes or []:
            path = f"/tmp/{meme.name}"
            with open(path, "wb") as f:
                f.write(meme.read())
            result = ingest_meme(path)
            summary["memes_ingested"] += result["memes_ingested"]
            summary["memes_deduped"] += result["memes_deduped"]

        for text_file in texts or []:
            path = f"/tmp/{text_file.name}"
            with open(path, "wb") as f:
                f.write(text_file.read())
            result = ingest_text(path)
            summary["evidence_added"] += result["evidence_added"]
            summary["claims_created"] += result["claims_created"]

        st.success("Ingestion completed")
        st.json(summary)

    if st.button("Run decay"):
        updated = apply_decay()
        st.info(f"Decay applied to {updated} claims")
