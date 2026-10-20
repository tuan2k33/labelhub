"""Smoke test of Auto mode: upload -> SAM3 autolabel -> accept -> versions -> YOLO export."""
import os
import sys
import tempfile
import time
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
os.environ["LABELHUB_DATA"] = str(tmp)

import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from labelhub import api

samples = [Path(p) for p in sys.argv[2:]]
classes = sys.argv[1].split(",")
out_dir = Path(os.environ.get("SMOKE_OUT", tmp))

c = TestClient(api.create_app())
c.put("/classes", json=classes)
imgs = c.post("/images", files=[("files", (p.name, p.read_bytes(), "image/jpeg")) for p in samples]).json()
print("uploaded", [(i["filename"], i["width"], i["height"]) for i in imgs])

t = time.time()
print("autolabel", c.post("/autolabel", json={}).json(), f"{time.time() - t:.1f}s incl. model load")
t = time.time()
print("autolabel again", c.post("/autolabel", json={"hashes": [imgs[0]["hash"]]}).json(), f"{time.time() - t:.1f}s")
print("peak vram MB", round(torch.cuda.max_memory_allocated() / 1e6))

for i in imgs:
    anns = c.get(f"/images/{i['hash']}/annotations").json()
    print(i["filename"], [(a["class_name"], round(a["confidence"], 2)) for a in anns])

print("accept", c.post("/proposals/accept", json={"min_confidence": 0.5}).json())
iv = c.post("/image-versions", json={"ratios": [1, 0, 0]}).json()
lv = c.post("/label-versions", json={"note": "auto sam3"}).json()
print("image version", iv["name"], iv["images"], "| label version", lv["name"], lv["images"], "images", lv["boxes"], "boxes")

import io, zipfile
z = zipfile.ZipFile(io.BytesIO(c.get("/export", params={"image_version": iv["name"], "label_version": lv["name"]}).content))
print("export", [n.split("/")[-1][:20] for n in z.namelist()][:8])
yaml = [n for n in z.namelist() if n.endswith("data.yaml")][0]
print(z.read(yaml).decode().split("names:")[1].strip())

for i in imgs:
    pil = Image.open(samples[[s.name for s in samples].index(i["filename"])]).convert("RGB")
    d = ImageDraw.Draw(pil)
    for a in c.get(f"/images/{i['hash']}/annotations").json():
        x1, y1 = (a["cx"] - a["w"] / 2) * pil.width, (a["cy"] - a["h"] / 2) * pil.height
        x2, y2 = (a["cx"] + a["w"] / 2) * pil.width, (a["cy"] + a["h"] / 2) * pil.height
        d.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        d.text((x1 + 4, y1 + 2), f"{a['class_name']} {a['confidence']:.2f}", fill="yellow")
    pil.save(out_dir / f"smoke_{i['filename']}")
