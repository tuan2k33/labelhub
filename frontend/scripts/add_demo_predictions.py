"""Run a trained model on the demo images and add the real detections and latency to data.json.

Usage: add_demo_predictions.py <demo data dir> <best.pt> <model name> <data.json> [imgsz]
"""
import json
import os
import sys
import time
from pathlib import Path

demo_dir, weights, model_name, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
imgsz = int(sys.argv[5]) if len(sys.argv) > 5 else 512
os.environ["LABELHUB_DATA"] = str(demo_dir)

from sqlmodel import Session, select
from ultralytics import YOLO

from labelhub import db

model = YOLO(weights)
engine = db.make_engine()
data = json.loads(out.read_text())
preds, warm_totals = {}, []
with Session(engine) as s:
    for im in s.exec(select(db.Image)).all():
        path = demo_dir / "images" / im.hash[:2] / f"{im.hash}.{im.ext}"
        model.predict(str(path), conf=0.05, imgsz=imgsz, device=0, verbose=False)  # warm-up, not counted
        totals, infers = [], []
        for _ in range(8):
            t = time.perf_counter()
            r = model.predict(str(path), conf=0.05, imgsz=imgsz, device=0, verbose=False)[0]
            totals.append((time.perf_counter() - t) * 1000)
            infers.append(r.speed["inference"])
        warm_totals += totals
        totals.sort(); infers.sort()
        preds[im.hash] = {
            "width": im.width, "height": im.height,
            "total_ms": round(totals[len(totals) // 2], 1), "inference_ms": round(infers[len(infers) // 2], 1),
            "detections": [
                {"class_name": r.names[int(c)], "confidence": round(float(sc), 4), "box": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}}
                for b, c, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
            ],
        }
warm_totals.sort()
stats = {
    "runs": len(warm_totals), "warm_runs": len(warm_totals), "device": "0",
    "p50_ms": round(warm_totals[len(warm_totals) // 2], 1), "p95_ms": round(warm_totals[int(len(warm_totals) * 0.95)], 1),
    "mean_ms": round(sum(warm_totals) / len(warm_totals), 1),
}
data["predictions"] = {"model": model_name, "imgsz": imgsz, "images": preds}
for m in data["models"]:
    if m["name"] == model_name:
        m["inference"] = stats
out.write_text(json.dumps(data))
print(len(preds), "images;", sum(len(p["detections"]) for p in preds.values()), "detections at conf>=0.05;", stats)
