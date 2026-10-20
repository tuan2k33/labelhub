"""Draw SAM3 (teacher) and the distilled YOLO (student) side by side on held-out test images.

Usage: compare_viz.py <out_dir> [n_images] [sam3_thr] [yolo_conf] [run_name]
"""
import json
import os
import random
import sys
from pathlib import Path

HOME = Path.home() / ".dev/data"
STATE = HOME / "lh_coco"
os.environ["LABELHUB_DATA"] = str(STATE)
CLASSES = ["person", "car", "chair", "bottle"]
COLORS = {"person": "#00e676", "car": "#ff9100", "chair": "#40c4ff", "bottle": "#ff4081"}

if __name__ == "__main__":
    from PIL import Image, ImageDraw
    from sqlmodel import Session, select

    from labelhub import db
    from labelhub.autolabel import Sam3Detector
    from ultralytics import YOLO

    out_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    sam_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    yolo_conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    run = sys.argv[5] if len(sys.argv) > 5 else "2026.10.1_2026.10.1"

    engine = db.make_engine()
    with Session(engine) as s:
        iv = s.get(db.ImageVersion, "2026.10.1")
        assign = json.loads(iv.split_json)["assign"]
        filenames = {i.hash: i.filename for i in s.exec(select(db.Image)).all()}
    export = STATE / "export/2026.10.1_2026.10.1"
    test = sorted(h for h, sp in assign.items() if sp == "test")
    picks = random.Random(1).sample(test, min(len(test), n * 4))
    paths = [next((export / "images/test").glob(f"{h}.*")) for h in picks]

    sam = Sam3Detector(threshold=0.3)
    yolo = YOLO(str(STATE / "runs" / run / "weights/best.pt"))

    def draw(img, dets, title):
        im = img.copy()
        d = ImageDraw.Draw(im)
        for name, (x1, y1, x2, y2), sc in dets:
            d.rectangle([x1, y1, x2, y2], outline=COLORS[name], width=3)
            d.text((x1 + 3, y1 + 2), f"{name} {sc:.2f}", fill=COLORS[name])
        d.rectangle([0, 0, 260, 18], fill="black")
        d.text((4, 3), f"{title}: {len(dets)} boxes", fill="white")
        return im

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for p in paths:
        if saved == n:
            break
        img = Image.open(p).convert("RGB")
        teacher = [(d.class_name, d.box, d.score) for d in sam.detect(img, CLASSES) if d.score >= sam_thr]
        r = yolo.predict(img, imgsz=512, conf=yolo_conf, verbose=False)[0]
        student = [(CLASSES[int(c)], tuple(b), s) for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        if not teacher and not student:
            continue
        left, right = draw(img, teacher, f"SAM3 >= {sam_thr}"), draw(img, student, f"YOLO26n >= {yolo_conf}")
        sheet = Image.new("RGB", (img.width * 2 + 10, img.height), "white")
        sheet.paste(left, (0, 0))
        sheet.paste(right, (img.width + 10, 0))
        sheet.save(out_dir / f"cmp_{Path(filenames[p.stem]).stem}.png")
        saved += 1
        print(filenames[p.stem], "SAM3", len(teacher), "YOLO", len(student))
