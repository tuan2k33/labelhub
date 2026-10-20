import hashlib
import io
from pathlib import Path

from PIL import Image as PILImage
from sqlmodel import Session

from . import config
from .db import Image


def image_path(hash: str, ext: str) -> Path:
    return config.images_dir() / hash[:2] / f"{hash}.{ext}"


def add_image(session: Session, data: bytes, filename: str) -> Image:
    h = hashlib.sha256(data).hexdigest()
    existing = session.get(Image, h)
    if existing:
        if existing.removed:
            existing.removed = False
            session.add(existing)
            session.commit()
            session.refresh(existing)
        return existing
    pil = PILImage.open(io.BytesIO(data))
    ext = (pil.format or "jpeg").lower().replace("jpeg", "jpg")
    path = image_path(h, ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    img = Image(hash=h, filename=filename, ext=ext, width=pil.width, height=pil.height)
    session.add(img)
    session.commit()
    session.refresh(img)
    return img
