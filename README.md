Project: Qdrant-Powered Multimodal Misinformation Correlation Engine (Memes + Articles + Social Claims)
Context / Hackathon Requirements (must satisfy)

Build an AI agent/application powered by Qdrant as the primary vector search engine to enable Search, Memory, and/or Recommendations for societal impact 

Qdrant - MAS PS Final - Convolv…

. The solution must demonstrate:

Effective multimodal retrieval and correct use of embeddings + similarity search, with meaningful payload/metadata filtering 

Qdrant - MAS PS Final - Convolv…

Memory beyond a single prompt, including evolving representations (updates/decay/reinforcement/deletion), and clear distinction between knowledge/context/history 

Qdrant - MAS PS Final - Convolv…

Societal relevance & responsibility (bias/privacy/safety/explainability, realistic assumptions) 

Qdrant - MAS PS Final - Convolv…

Evidence-based outputs: grounded responses, traceable reasoning paths showing what was retrieved and why, avoiding hallucination 

Qdrant - MAS PS Final - Convolv…

Deliverables must include reproducible code, setup instructions, documentation/report (≤10 pages excluding appendix), and demo examples/logs/screenshots optional 

Qdrant - MAS PS Final - Convolv…

. Evaluation prioritizes meaningful Qdrant usage, retrieval/memory quality, societal impact, robustness, and clear documentation 

Qdrant - MAS PS Final - Convolv…

.

1) Your Chosen Problem & Scope (lock this)
Problem Statement (Digital trust / misinformation)

Create a system that correlates misinformation claims across:

Memes (images)

Articles (text)

Social-media style claims (text)

Focus is not healthcare; keep it general digital trust/misinformation. The system should help users understand:

Whether a meme/claim resembles previously seen claims (paraphrases, variants)

How the claim spread over time (timeline)

What evidence in the indexed corpus supports or contradicts the claim

Similar meme variants (same claim in different templates)

Constraints (must follow)

Qdrant must be the core memory + retrieval substrate (not optional, not just storage).

No paid APIs: no OpenAI/Gemini/Anthropic, etc.

Use only free/local components and open-source models.

Modalities: TEXT + IMAGE only (memes + text). No audio/video.

UI: Streamlit minimal UI (not UI-heavy, but enough to demo).

System outputs must be evidence-grounded with a visible trace.

2) System Overview (what you are building)

A claim-centric engine that:

Ingests memes and text sources, extracts “claim units” and “evidence snippets”

Embeds and indexes them in Qdrant (multimodal)

Supports queries by meme image or text claim

Returns:

matched canonical claim clusters

similar meme variants

evidence snippets grouped by stance (support/contradict/mention)

timeline of appearances

trace panel: what was retrieved, scores, filters, why it influenced output

Maintains long-term memory via:

canonicalization/merging of claims

reinforcement, contradiction, decay

update logs (audit trail)

Important: This is a correlation + evidence system, not a “truth oracle.” It should avoid asserting truth unless supported by retrieved evidence. If evidence is insufficient, it must say so.

3) Architecture (modules + responsibilities)

Implement as a Python project with the following structure:

app/

streamlit_app.py: Streamlit UI with 3 pages:

Analyze Meme (image upload)

Analyze Claim/Text (text input)

Ingest Corpus (admin ingestion of meme images + text files)

core/

config.py: env vars + defaults

schemas.py: Pydantic models for Claim, EvidenceSnippet, MemeMedia, RetrievalTrace, Response

utils.py: helpers (time, hashing, cleaning)

models/ (free/local)

text_embedder.py: sentence-transformers embedder

image_embedder.py: CLIP image embedder

ocr.py: Tesseract OCR wrapper + preprocessing

claim_extractor.py: rule-based claim extraction; optional Ollama if enabled

stance_classifier.py: rule-based stance; optional Ollama if enabled

qdrant_store/

client.py: init Qdrant client (local or cloud)

collections.py: create/update collections and indexes

crud.py: upsert/search/update payloads

ingestion/

ingest_meme.py: meme → OCR → claim candidates → embeddings → upsert to Qdrant + linkages

ingest_text.py: article/social text → chunk → claim candidates → evidence snippets → embeddings → upserts

dedup.py: meme dedup with pHash; text dedup with hashing

memory/

canonicalize.py: merge new claim into canonical claim cluster via similarity threshold

confidence.py: reinforcement/contradiction scoring

decay.py: optional scheduled decay function

events.py: log memory updates to SQLite for auditing

storage/

sqlite.py: store raw sources, mappings, event logs

files.py: store uploaded memes/text locally with deterministic IDs

docs/

README.md

report_outline.md (≤10 pages plan)

architecture_diagram.png or ASCII diagram

4) Models & Embeddings (no paid APIs)

All default dependencies must be free and run locally.

Text embeddings (choose one; configurable)

Default: sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast)

Option: intfloat/e5-base-v2 (768-dim, stronger but heavier)

Use cosine similarity.

Image embeddings

CLIP via HuggingFace transformers: openai/clip-vit-base-patch32 (512-dim)

OCR

Tesseract via pytesseract (free local). Provide Windows/Linux install steps in README.

Claim extraction (must not require paid LLM)

Implement Hybrid:

Must work with rule-based extraction only

Optional improvement: Ollama local (still free) for better claim extraction/stance classification, toggled by env flag.

If Ollama is absent, system remains fully functional.

5) Qdrant Design (Collections + payload schema)

Qdrant must be used as primary retrieval and memory system 

Qdrant - MAS PS Final - Convolv…

.

Collection A: claims (canonical claims / long-term memory)

Point = one canonical claim cluster.

Vectors

text_dense: dim = text_dim (384 or 768), distance = cosine

Payload (minimum)

canonical_claim_id (string UUID)

claim_text (string)

first_seen_ts (ISO string)

last_seen_ts (ISO string)

mention_count (int)

source_types (list: meme/article/social)

support_count (int)

contradict_count (int)

confidence (float 0..1)

status (enum: unverified/disputed/likely_false/likely_true)

linked_evidence_ids (list) OR store evidence by filter on claim_id

linked_media_ids (list) OR store media via filter on linked_claim_ids

optional: language, entities, topics

Collection B: evidence_snippets (grounding evidence)

Point = one snippet that supports/contradicts/mentions a claim.

Vectors

snippet_dense: dim = text_dim, cosine

Payload

evidence_id

claim_id (canonical_claim_id)

snippet_text

stance (support/contradict/mention)

source_id

source_type (article/social)

timestamp

url (optional)

credibility_tier (A/B/C heuristic)

Collection C: media_memes (memes + multimodal retrieval)

Point = one meme image.

Vectors

image_dense: dim=512, cosine

ocr_text_dense: dim=text_dim, cosine

Payload

media_id

source_id

timestamp

phash (dedup)

ocr_text

linked_claim_ids (list of canonical_claim_id)

optional template_cluster_id

Must-have Qdrant features in use

Use metadata filtering (payload filters) meaningfully (time range, source_type, etc.) 

Qdrant - MAS PS Final - Convolv…

Use upserts and payload updates to represent evolving memory 

Qdrant - MAS PS Final - Convolv…

Provide traceable retrieval path details 

Qdrant - MAS PS Final - Convolv…

6) Retrieval & Reasoning (evidence-based, traceable)
Query Type 1: Meme Image

Steps:

OCR → cleaned text

Compute embeddings:

image_dense from CLIP

ocr_text_dense from text model

Retrieve:

Search media_memes by image_dense topK_img (meme variants)

Search claims by embedding(OCR text) topK_claim

Merge candidate claims:

top claim hits

plus claim IDs linked to top meme variants

Expand evidence:

Fetch evidence snippets by claim_id filter

Pull both support and contradict (if available)

Build timeline:

Use first_seen_ts, last_seen_ts, and source timestamps in SQLite

Output must include:

Top matched claims with similarity scores and retrieval path label (OCR/text vs image variant)

Similar meme variants (render images)

Evidence snippets grouped by stance

Timeline table

Trace panel

Query Type 2: Text Claim

Steps:

Embed query text

Search claims topK

Expand evidence + linked meme variants

Output same style with trace.

Evidence-grounded response rules

No claims about reality unless supported by retrieved evidence.

If evidence is insufficient or conflicting, label as “insufficient/conflicting evidence in corpus”.

Always show “why”: claim IDs, evidence IDs, similarity scores, filters.

7) Long-term Memory (beyond one prompt)

Must implement evolving memory logic 

Qdrant - MAS PS Final - Convolv…

.

Canonicalization (merge vs create)

When ingesting a claim candidate:

Search claims top5

If max similarity ≥ threshold (default 0.85, configurable):

Merge into that canonical claim:

update mention_count += 1

update last_seen_ts = now

add source_type if new

Link source_id ↔ canonical_claim_id in SQLite

Else:

Create new canonical claim with new UUID

mention_count=1, first_seen_ts=now, last_seen_ts=now, confidence=0.5, status=unverified

Confidence / reinforcement / contradiction

Implement a transparent heuristic:

Start 0.5 for new claims

+0.05 per independent source mention (cap)

+0.10 if credible tier A

−0.15 if contradiction from tier A

Decay toward 0.5 if not seen in N days (optional; implement as function callable from admin page)

Log every update to SQLite events table:

time, canonical_claim_id, event_type, delta, reason, source_id

8) Streamlit Minimal UI (must implement)
Page 1: Analyze Meme

Components:

Image uploader

Display meme

Display OCR text

Display extracted claim candidates

Results:

Top matched canonical claims (table with score + last_seen)

Similar meme variants (grid of images with similarity)

Evidence snippets grouped: Support / Contradict / Mention

Timeline table

“Why this result?” expander:

Retrieval trace table: (collection, point_id, score, path, filters used, payload preview)

Page 2: Analyze Claim/Text

Components:

Text area

Results same as above (minus image variants optional but recommended)

Page 3: Ingest Corpus (Admin)

Components:

Upload multiple meme images

Upload multiple text files (articles / social claim dumps)

Optional: paste text

Ingest button + progress indicator

Summary counters:

memes ingested, memes deduped

new canonical claims created

claims merged

evidence snippets added

Optional “Run decay” button (calls decay function and logs updates)

9) Setup & Running (must be included in README)
Local Qdrant (default)

Docker command:

Expose 6333 and 6334

Provide health check step

Python dependencies

Provide requirements.txt including at least:

streamlit

qdrant-client

sentence-transformers

transformers

torch

pillow

pytesseract

opencv-python (optional)

imagehash

numpy

pandas

pydantic

python-dotenv

tqdm

requests

beautifulsoup4 (optional)

OCR setup

Windows: install Tesseract, set TESSERACT_CMD

Linux: install via package manager

Provide a quick test command inside repo

Run app

streamlit run app/streamlit_app.py

10) API keys / secrets handling (must be flexible)

Default local mode should require no API keys.

Environment variables (provide .env.example):

QDRANT_URL (default http://localhost:6333)

QDRANT_API_KEY (optional; for Qdrant Cloud if organizers provide)

TEXT_MODEL_NAME (default MiniLM)

IMAGE_MODEL_NAME (default CLIP ViT-B/32)

USE_OLLAMA (false by default)

OLLAMA_MODEL (optional)

TESSERACT_CMD (optional, Windows)

DATA_DIR (where uploaded files live)

SQLITE_PATH

Notes:

Qdrant local: no key

Qdrant cloud: accept API key if provided by hackathon organizers

No other paid keys allowed.

11) Documentation/report outline (≤10 pages)

Include:

Problem statement: misinformation & digital trust + why it matters

System design: architecture diagram, components

Why Qdrant is critical (vector search + payload filters + low-latency + updates)

Multimodal strategy: meme OCR + image embeddings + text embeddings

Search/memory logic: canonicalization, updates, decay, contradiction, traceability

Limitations & ethics: dataset bias, OCR errors, false correlations, privacy, safe messaging

Demo examples: screenshots, sample queries, trace panel

12) Build Output Requirements (what the coder must produce)

The coder must output:

Full folder structure + all Python files

requirements.txt

.env.example

README with setup + demo steps

Auto-create Qdrant collections if missing

Include small sample folders: data/memes/, data/text/ (placeholders ok)

Ensure everything runs end-to-end locally with Docker + Python only

13) Non-negotiable acceptance tests

The implementation is “done” only when:

You can ingest 10+ memes and 10+ text docs

Meme query returns:

similar memes (image similarity)

matched claims (OCR text similarity)

evidence snippets retrieved from Qdrant

trace panel with IDs + scores

Text query returns matched claims + evidence + trace

Ingesting more data updates claim memory (mention_count, last_seen_ts, confidence) and logs events

No paid API usage anywhere
