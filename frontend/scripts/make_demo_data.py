"""Collect the data for the static demo: sample images + real SAM3 proposals from the demo DB,
and real versions/models from a LabelHub server that ran the COCO experiment.

Usage: make_demo_data.py <demo data dir> <server url with real versions> <out json>
"""
import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image as PILImage
from sqlmodel import Session, select

demo_dir, server, out = Path(sys.argv[1]), sys.argv[2].rstrip("/"), Path(sys.argv[3])

import os
os.environ["LABELHUB_DATA"] = str(demo_dir)
from labelhub import db  # noqa: E402


def get(path):
    with urllib.request.urlopen(server + path, timeout=30) as r:
        return json.load(r)


engine = db.make_engine()
images, classes = [], []
with Session(engine) as s:
    classes = [c.name for c in s.exec(select(db.WsClass).order_by(db.WsClass.position)).all()]
    for im in s.exec(select(db.Image).order_by(db.Image.filename)).all():
        path = demo_dir / "images" / im.hash[:2] / f"{im.hash}.{im.ext}"
        pil = PILImage.open(path).convert("RGB")
        pil.thumbnail((640, 640))
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=72)
        boxes = [
            {"class_name": a.class_name, "cx": a.cx, "cy": a.cy, "w": a.w, "h": a.h, "confidence": round(a.confidence, 3)}
            for a in s.exec(select(db.Annotation).where(db.Annotation.image_hash == im.hash, db.Annotation.source == "auto:sam3")).all()
        ]
        images.append({
            "hash": im.hash, "filename": im.filename, "width": im.width, "height": im.height,
            "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "boxes": boxes,
        })

# four images start un-labelled so the Auto-label button has something to do
candidates = [i for i in images if len(i["boxes"]) >= 3]
hidden = {i["hash"] for i in candidates[-4:]}
for i in images:
    i["pending_autolabel"] = i["hash"] in hidden

image_versions = get("/image-versions")
label_versions = get("/label-versions")
models = get("/models")
for m in models:
    if m["status"] == "done":
        m["log"] = get(f"/models/{m['name']}/log?tail=40")["log"]
diffs = {}
names = [v["name"] for v in label_versions]
for a in names:
    for b in names:
        if a < b:
            d = get(f"/label-versions/{a}/diff/{b}")
            diffs[f"{a}|{b}"] = {k: len(v) if isinstance(v, list) else v for k, v in d.items()}

out.write_text(json.dumps({
    "classes": classes, "images": images,
    "image_versions": image_versions, "label_versions": label_versions, "models": models, "label_diffs": diffs,
}))
print(len(images), "images;", sum(len(i["boxes"]) for i in images), "boxes;", len(hidden), "start un-labelled;",
      len(image_versions), "image versions;", len(label_versions), "label versions;", len(models), "models;",
      round(out.stat().st_size / 1e6, 2), "MB")
