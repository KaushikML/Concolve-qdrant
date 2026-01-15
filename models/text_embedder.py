from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import settings


class TextEmbedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.text_model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype="float32")


_embedder = None


def get_text_embedder() -> TextEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedder()
    return _embedder
