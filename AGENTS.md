# AGENTS.md

## Project overview
- spider search : multimodal classifier is a Streamlit app that correlates claims and evidence using Qdrant and optional Ollama.

## Setup
- Create a virtualenv and install dependencies: `python -m venv .venv` and `pip install -r requirements.txt`.
- Copy the env template: `cp .env.example .env`.

## Services
- Qdrant (required): `docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant`.
- Ollama (optional): `ollama serve` then `ollama pull llama3`.

## Run
- `streamlit run streamlit_app.py`

## Data and storage
- Qdrant data is stored in `qdrant_storage/`.
- SQLite metadata defaults to `data/app.db` (see `SQLITE_PATH` in `.env`).

## Tests
- No automated test suite in this repo.

## Useful entry points
- UI and workflows: `streamlit_app.py`.
- Ollama deduction prompt: `models/llm_reasoner.py`.
- Stance classification: `models/stance_classifier.py`.
