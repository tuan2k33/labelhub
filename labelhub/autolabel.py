from dataclasses import dataclass
from typing import Protocol

from PIL import Image as PILImage
from sqlmodel import Session, select

from .db import Annotation, Image
from .store import image_path


@dataclass
class Detection:
    class_name: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float


class Detector(Protocol):
    def detect(self, image: PILImage.Image, classes: list[str]) -> list[Detection]: ...


class Sam3Detector:
    def __init__(self, model_id: str = "facebook/sam3", threshold: float = 0.3):
        import torch
        from dotenv import load_dotenv
        from transformers import Sam3Model, Sam3Processor

        from .config import ROOT

        load_dotenv(ROOT / ".env")
        self.torch, self.threshold, self.dtype = torch, threshold, torch.float16
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=self.dtype).to("cuda").eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    def close(self) -> None:
        """Give the GPU memory back (training needs it on a 4 GB card)."""
        import gc

        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()

    def detect(self, image: PILImage.Image, classes: list[str]) -> list[Detection]:
        torch = self.torch
        img = self.processor(images=image, return_tensors="pt").to("cuda")
        sizes = img["original_sizes"].tolist()
        out: list[Detection] = []
        with torch.no_grad():
            vision = self.model.get_vision_features(pixel_values=img["pixel_values"].to(self.dtype))
            for name in classes:
                txt = self.processor(text=name, return_tensors="pt").to("cuda")
                res = self.model(vision_embeds=vision, input_ids=txt["input_ids"], attention_mask=txt["attention_mask"])
                r = self.processor.post_process_instance_segmentation(
                    res, threshold=self.threshold, mask_threshold=0.5, target_sizes=sizes
                )[0]
                out += [Detection(name, tuple(b), s) for b, s in zip(r["boxes"].tolist(), r["scores"].tolist())]
        return out


def autolabel(
    session: Session,
    detector: Detector,
    hashes: list[str],
    classes: list[str],
    source: str = "auto:sam3",
    min_confidence: float | None = None,
) -> int:
    """Replace each image's earlier boxes from this model with fresh ones (boxes drawn by hand stay). Returns boxes added.

    With min_confidence, boxes at or above it are accepted straight away and the rest are kept
    hidden as rejected (so a lower threshold can bring them back); without it all stay proposed."""
    added = 0
    for h in hashes:
        img = session.get(Image, h)
        for old in session.exec(select(Annotation).where(Annotation.image_hash == h, Annotation.source == source)).all():
            session.delete(old)
        with PILImage.open(image_path(img.hash, img.ext)) as pil:
            dets = detector.detect(pil.convert("RGB"), classes)
        img.auto_done = True
        session.add(img)
        for d in dets:
            x1, y1, x2, y2 = d.box
            session.add(
                Annotation(
                    image_hash=h,
                    class_name=d.class_name,
                    cx=(x1 + x2) / 2 / img.width,
                    cy=(y1 + y2) / 2 / img.height,
                    w=(x2 - x1) / img.width,
                    h=(y2 - y1) / img.height,
                    source=source,
                    status="proposed" if min_confidence is None else ("accepted" if d.score >= min_confidence else "rejected"),
                    confidence=d.score,
                )
            )
            added += 1
    session.commit()
    return added


def accept_proposals(session: Session, hashes: list[str], min_confidence: float, mark_reviewed: bool = True) -> dict:
    """Auto mode: auto boxes >= min_confidence become accepted, the rest rejected (kept, so the
    threshold can be changed later). Marks images reviewed."""
    accepted = rejected = 0
    for h in hashes:
        for a in session.exec(
            select(Annotation).where(Annotation.image_hash == h, Annotation.source.startswith("auto:"))
        ).all():
            if a.confidence is None:
                continue
            a.status = "accepted" if a.confidence >= min_confidence else "rejected"
            session.add(a)
            accepted += a.status == "accepted"
            rejected += a.status == "rejected"
        if mark_reviewed:
            img = session.get(Image, h)
            img.reviewed = True
            session.add(img)
    session.commit()
    return {"accepted": accepted, "rejected": rejected, "images": len(hashes)}


def view_image(session: Session, h: str, min_confidence: float = 0.5) -> Image:
    """Opening an image counts as reviewing it: pending proposals are settled and it is marked reviewed."""
    for a in session.exec(select(Annotation).where(Annotation.image_hash == h, Annotation.status == "proposed")).all():
        a.status = "accepted" if (a.confidence is None or a.confidence >= min_confidence) else "rejected"
        session.add(a)
    img = session.get(Image, h)
    img.reviewed = True
    session.add(img)
    session.commit()
    session.refresh(img)
    return img
