from typing import Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract

from core.config import settings


if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def preprocess(image: Image.Image) -> Image.Image:
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return Image.fromarray(img)


def extract_text(image: Image.Image, lang: str = "eng") -> str:
    processed = preprocess(image)
    text = pytesseract.image_to_string(processed, lang=lang)
    return text.strip()
