from typing import Optional

import imagehash
from PIL import Image

from core.utils import sha256_text


def meme_phash(image: Image.Image) -> str:
    return str(imagehash.phash(image))


def text_hash(text: str) -> str:
    return sha256_text(text)


def is_similar_phash(phash_a: str, phash_b: str, threshold: int = 5) -> bool:
    return imagehash.hex_to_hash(phash_a) - imagehash.hex_to_hash(phash_b) <= threshold
