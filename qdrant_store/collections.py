from qdrant_client.http import models

from core.config import settings
from models.text_embedder import get_text_embedder
from qdrant_store.client import get_client


CLAIMS_COLLECTION = "claims"
EVIDENCE_COLLECTION = "evidence_snippets"
MEDIA_COLLECTION = "media_memes"


def ensure_collections() -> None:
    client = get_client()
    collections = {col.name for col in client.get_collections().collections}

    text_dim = get_text_embedder().embed(["dim check"]).shape[1]

    if CLAIMS_COLLECTION not in collections:
        client.create_collection(
            collection_name=CLAIMS_COLLECTION,
            vectors_config={
                "text_dense": models.VectorParams(size=text_dim, distance=models.Distance.COSINE)
            },
        )

    if EVIDENCE_COLLECTION not in collections:
        client.create_collection(
            collection_name=EVIDENCE_COLLECTION,
            vectors_config={
                "snippet_dense": models.VectorParams(size=text_dim, distance=models.Distance.COSINE)
            },
        )

    if MEDIA_COLLECTION not in collections:
        client.create_collection(
            collection_name=MEDIA_COLLECTION,
            vectors_config={
                "image_dense": models.VectorParams(size=512, distance=models.Distance.COSINE),
                "ocr_text_dense": models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
            },
        )
