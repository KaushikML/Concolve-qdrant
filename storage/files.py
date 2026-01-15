import os
import shutil
from typing import BinaryIO

from core.config import settings
from core.utils import sha256_text


def ensure_data_dirs() -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.data_dir, "memes"), exist_ok=True)
    os.makedirs(os.path.join(settings.data_dir, "text"), exist_ok=True)


def save_uploaded_file(file_obj: BinaryIO, suffix: str) -> str:
    ensure_data_dirs()
    content = file_obj.read()
    file_hash = sha256_text(content.hex())
    filename = f"{file_hash}{suffix}"
    path = os.path.join(settings.data_dir, "uploads", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


def save_meme_file(path: str) -> str:
    ensure_data_dirs()
    filename = os.path.basename(path)
    dest = os.path.join(settings.data_dir, "memes", filename)
    shutil.copyfile(path, dest)
    return dest


def save_text_file(path: str) -> str:
    ensure_data_dirs()
    filename = os.path.basename(path)
    dest = os.path.join(settings.data_dir, "text", filename)
    shutil.copyfile(path, dest)
    return dest
