import threading
import time
from pathlib import Path

from PIL import Image as PILImage
from sqlmodel import Session, select

from .db import InferenceLog

DEVICE: str | None = None  # tests force "cpu"
_models: dict[str, object] = {}
_lock = threading.Lock()
MAX_LOADED = 3


def pick_device(busy: bool) -> str:
    """GPU unless a training run needs its memory (or there is no GPU)."""
    if DEVICE:
        return DEVICE
    import torch

    return "cpu" if busy or not torch.cuda.is_available() else "0"


def _load(path: Path):
    key = str(path)
    if key in _models:
        return _models[key], False
    from ultralytics import YOLO

    while len(_models) >= MAX_LOADED:
        _models.pop(next(iter(_models)))
    _models[key] = YOLO(key)
    return _models[key], True


def predict(session: Session, name: str, weights: Path, image: PILImage.Image, conf: float, imgsz: int, device: str, source: str = "ui") -> dict:
    with _lock:
        model, cold = _load(weights)
        t0 = time.perf_counter()
        r = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        total = (time.perf_counter() - t0) * 1000
    names = r.names
    dets = [
        {"class_name": names[int(c)], "confidence": round(float(sc), 4), "box": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}}
        for b, c, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
    ]
    infer = float(r.speed.get("inference", 0.0))
    session.add(InferenceLog(model=name, total_ms=total, infer_ms=infer, device=device, cold=cold, boxes=len(dets), width=image.width, height=image.height, source=source))
    session.commit()
    return {
        "model": name, "width": image.width, "height": image.height, "device": device, "imgsz": imgsz,
        "total_ms": round(total, 1), "inference_ms": round(infer, 1), "cold": cold, "detections": dets,
    }


def _pct(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def stats(session: Session, names: list[str] | None = None) -> dict[str, dict]:
    """Latency per model. The first call after a model is loaded is slow (warm-up) and left out."""
    q = select(InferenceLog)
    if names is not None:
        q = q.where(InferenceLog.model.in_(names))
    rows: dict[str, list[InferenceLog]] = {}
    for r in session.exec(q).all():
        rows.setdefault(r.model, []).append(r)
    out = {}
    for name, logs in rows.items():
        warm = [x.total_ms for x in logs if not x.cold]
        out[name] = {
            "runs": len(logs),
            "warm_runs": len(warm),
            "p50_ms": round(_pct(warm, 0.5), 1) if warm else None,
            "p95_ms": round(_pct(warm, 0.95), 1) if warm else None,
            "mean_ms": round(sum(warm) / len(warm), 1) if warm else None,
            "device": logs[-1].device,
        }
    return out
