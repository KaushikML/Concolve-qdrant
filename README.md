# Concolve: Qdrant-Powered Multimodal Misinformation Correlation Engine

Concolve is a claim-centric, multimodal (text + image) system that correlates misinformation across memes, articles, and social-media style claims. It uses **Qdrant** as the primary vector memory and retrieval engine, providing evidence-grounded responses, traceability, and long-term memory updates.

## Why this matters
Meme-based misinformation spreads fast, often with text paraphrases and image variants. Concolve helps analysts and journalists surface:

- Related claim clusters (paraphrases/variants)
- Similar meme variants (image similarity)
- Supporting/contradicting evidence snippets
- A traceable retrieval path (scores, filters, IDs)

> Concolve is a correlation and evidence system — **not a truth oracle**. It reports what the indexed corpus supports and flags insufficient/conflicting evidence when needed.

---

## Architecture (High Level)

```
[Streamlit UI]
   |-- Analyze Meme (Image Upload)
   |-- Analyze Claim/Text
   |-- Ingest Corpus
        | OCR / Embeddings / Claim Extraction
        v
 [Ingestion Pipeline] -----> [Qdrant Collections]
        |                      - claims (text_dense)
        |                      - evidence_snippets (snippet_dense)
        |                      - media_memes (image_dense + ocr_text_dense)
        v
 [SQLite Audit + Mappings]
        |
        v
 [Retrieval + Trace Panel]
```

See `docs/architecture_diagram.txt` for the ASCII diagram and `docs/report_outline.md` for the report outline.

---

## Features

- **Multimodal retrieval**: CLIP image embeddings + OCR text embeddings for memes.
- **Qdrant-centric memory**: canonical claims and evidence stored in Qdrant collections with metadata filtering.
- **Long-term memory**: claim canonicalization, reinforcement, contradiction updates, decay, and audit logs in SQLite.
- **Evidence-based responses**: stance-grouped evidence snippets with visible trace panel.

---

## Repo Structure

```
app/                  # Streamlit UI
core/                 # config, schemas, utils
models/               # text/image embedding, OCR, rule-based extraction
qdrant_store/         # Qdrant client + CRUD helpers
ingestion/            # meme/text ingestion pipelines
memory/               # canonicalization, confidence, decay, events
storage/              # SQLite + file storage helpers
docs/                 # report outline + architecture diagram
```

---

## Setup

### 1) Start Qdrant (Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Health check:

```bash
curl http://localhost:6333/collections
```

### 2) Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) OCR setup (Tesseract)

- **Linux**: `sudo apt-get install tesseract-ocr`
- **Windows**: install from the official installer and set `TESSERACT_CMD` in `.env`.

### 4) Configure environment

Copy `.env.example` to `.env` and adjust if needed:

```bash
cp .env.example .env
```

### 5) Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## Quick Demo Flow

1. **Ingest Corpus**
   - Upload 10+ meme images and 10+ text documents (TXT files).
   - System extracts claims, embeds, and stores them in Qdrant.

2. **Analyze Meme**
   - Upload a meme and view:
     - OCR text
     - Claim candidates
     - Similar meme variants (image similarity)
     - Matched claims + evidence snippets
     - Trace panel (IDs + scores)

3. **Analyze Claim/Text**
   - Enter a claim sentence.
   - View matched claims + evidence, plus trace panel.

4. **Run Decay** (optional)
   - Admin button triggers memory decay and logs updates.

---

## Qdrant Collections

### A) `claims`
- **Vector**: `text_dense` (384 or 768 dim, cosine)
- **Payload**: canonical claim, counts, timestamps, confidence, status, links

### B) `evidence_snippets`
- **Vector**: `snippet_dense` (text embedding)
- **Payload**: snippet text, stance, source metadata

### C) `media_memes`
- **Vectors**: `image_dense` (CLIP) + `ocr_text_dense`
- **Payload**: OCR text, pHash, linked claim IDs

---

## Evidence-Grounded Output Rules

- If evidence is insufficient or conflicting, the UI signals **insufficient/conflicting evidence**.
- Retrieval traces show collection, point IDs, similarity scores, and payload previews.

---

## Limitations & Ethics

- OCR errors can lead to imperfect claim extraction.
- Meme templates might bias similarity results.
- No personal data should be ingested without consent.
- Concolve does not assert truth — it surfaces evidence and uncertainties.

---

## Notes

- No paid APIs are used.
- Optional enhancement: enable Ollama locally by setting `USE_OLLAMA=true`.
