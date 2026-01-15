from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from core.config import settings


class ImageEmbedder:
    def __init__(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(settings.image_model_name)
        self.model = CLIPModel.from_pretrained(settings.image_model_name)
        self.model.eval()

    def embed(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        embeddings = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return embeddings.cpu().numpy().astype("float32")


_embedder = None


def get_image_embedder() -> ImageEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ImageEmbedder()
    return _embedder
