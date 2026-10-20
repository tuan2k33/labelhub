"""SAM3 -> YOLO distillation experiment on COCO val2017, run through the LabelHub pipeline.

Stages (run in order, each resumable):
  prep     pick N images with target classes, upload them to LabelHub
  label    SAM3 auto-label (proposals); GPU
  teacher  dump all proposals + score SAM3 vs COCO ground truth
  cross    score every trained run on the same export's test split
  train    accept >= thr, make versions, export YOLO, fine-tune yolo26n (pretrained, NMS-free), score student vs COCO GT; GPU

Usage: coco_distill.py <stage> [--n 1000] [--thr 0.5] [--epochs 80] [--imgsz 512]
"""
import argparse
import contextlib
import io
import json
import os
import random
import time
from pathlib import Path

HOME = Path.home() / ".dev/data"
STATE = HOME / "lh_coco"
COCO_DIR = HOME / "coco"
os.environ["LABELHUB_DATA"] = str(STATE)
CLASSES = {"person": 1, "car": 3, "chair": 62, "bottle": 44}
CAT_IDS = list(CLASSES.values())

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from labelhub import api, db
from labelhub.db import Annotation, Image

ap = argparse.ArgumentParser()
ap.add_argument("stage")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--imgsz", type=int, default=512)
ap.add_argument("--export", default=None, help="export dir name for the cross stage (default: newest)")
args = ap.parse_args()

engine = db.make_engine()
client = TestClient(api.create_app(engine))
ANN_FILE = COCO_DIR / "annotations/instances_val2017.json"
coco_ann = json.load(open(ANN_FILE)) if ANN_FILE.exists() else None


def image_id(filename: str) -> int:
    return int(Path(filename).stem)


def coco_eval(dets: list[dict], img_ids: list[int]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ids = set(img_ids)
    gt = COCO()
    gt.dataset = {
        "images": [{"id": i} for i in ids],
        "annotations": [a for a in coco_ann["annotations"] if a["image_id"] in ids and a["category_id"] in CAT_IDS],
        "categories": [{"id": c} for c in CAT_IDS],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        if not dets:
            return {"AP": 0.0, "AP50": 0.0, "per_class_AP50": {}}
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.imgIds = sorted(ids)
        ev.params.catIds = CAT_IDS
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    per = {}
    for k, name in enumerate(CLASSES):
        p = ev.eval["precision"][0, :, k, 0, 2]
        per[name] = round(float(p[p > -1].mean()), 3) if (p > -1).any() else 0.0
    return {"AP": round(ev.stats[0], 3), "AP50": round(ev.stats[1], 3), "per_class_AP50": per}


def load_images() -> dict[str, Image]:
    with Session(engine) as s:
        return {i.hash: i for i in s.exec(select(Image).where(Image.removed == False)).all()}  # noqa: E712


if __name__ == "__main__":
    if args.stage == "prep":
        cats = set(CAT_IDS)
        if coco_ann:
            has = sorted({a["image_id"] for a in coco_ann["annotations"] if a["category_id"] in cats and not a["iscrowd"]})
        else:
            has = sorted(int(p.stem) for p in (COCO_DIR / "val2017").glob("*.jpg"))
        random.Random(0).shuffle(has)
        ids = has[: args.n]
        client.put("/classes", json=list(CLASSES))
        for i in range(0, len(ids), 50):
            files = [("files", (f"{n:012d}.jpg", (COCO_DIR / f"val2017/{n:012d}.jpg").read_bytes(), "image/jpeg")) for n in ids[i : i + 50]]
            client.post("/images", files=files)
            print("uploaded", min(i + 50, len(ids)), flush=True)

    elif args.stage == "label":
        hashes = [h for h, im in load_images().items() if not im.reviewed]
        with Session(engine) as s:
            done = {a.image_hash for a in s.exec(select(Annotation).where(Annotation.source == "auto:sam3")).all()}
        t0 = time.time()
        todo = [h for h in hashes if h not in done]
        print(f"{len(todo)} images to label (SAM3 loads on first call)", flush=True)
        for i in range(0, len(todo), 50):
            r = client.post("/autolabel", json={"hashes": todo[i : i + 50]}).json()
            print(f"{min(i + 50, len(todo))}/{len(todo)} boxes+{r['boxes']} elapsed {time.time() - t0:.0f}s", flush=True)

    elif args.stage == "teacher":
        imgs = load_images()
        dets = []
        with Session(engine) as s:
            for a in s.exec(select(Annotation).where(Annotation.status == "proposed")).all():
                im = imgs[a.image_hash]
                dets.append(
                    {
                        "image_id": image_id(im.filename),
                        "category_id": CLASSES[a.class_name],
                        "bbox": [(a.cx - a.w / 2) * im.width, (a.cy - a.h / 2) * im.height, a.w * im.width, a.h * im.height],
                        "score": a.confidence,
                    }
                )
        (STATE / "teacher_dets.json").write_text(json.dumps(dets))
        ids = [image_id(i.filename) for i in imgs.values()]
        print(f"{len(dets)} proposals on {len(ids)} images")
        for name in CLASSES:
            sc = [d["score"] for d in dets if d["category_id"] == CLASSES[name]]
            print(f"  {name}: {len(sc)} proposals, {sum(x >= args.thr for x in sc)} with score>={args.thr}")
        for thr in (0.0, 0.3, 0.5, 0.7) if coco_ann else ():
            r = coco_eval([d for d in dets if d["score"] >= thr], ids)
            print(f"SAM3 vs COCO GT (all images), score>={thr}: {r}")

    elif args.stage == "train":
        from ultralytics import YOLO

        from labelhub import versions

        imgs = load_images()
        print("accept", client.post("/proposals/accept", json={"min_confidence": args.thr}).json())
        iv = client.post("/image-versions", json={"ratios": [0.7, 0.1, 0.2], "seed": 0}).json()
        if "name" not in iv:
            iv = client.get("/image-versions").json()[-1]
        lv = client.post("/label-versions", json={"note": f"sam3 >= {args.thr}"}).json()
        if "name" not in lv:
            lv = client.get("/label-versions").json()[-1]
        print("versions", iv["name"], iv["split"], "|", lv["name"], lv["images"], "images", lv["boxes"], "boxes")
        with Session(engine) as s:
            ivo = s.get(db.ImageVersion, iv["name"])
            lvo = s.get(db.LabelVersion, lv["name"])
            out = versions.export_yolo(s, ivo, lvo, STATE / "export" / f"{iv['name']}_{lv['name']}")
            assign = json.loads(ivo.split_json)["assign"]
        test_hashes = [h for h, sp in assign.items() if sp == "test"]

        t0 = time.time()
        model = YOLO(str(STATE / "yolo26n.pt"))
        model.train(
            data=str(out / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=16, workers=4,
            project=str(STATE / "runs"), name=f"{iv['name']}_{lv['name']}", exist_ok=True, plots=False,
        )
        print(f"train time {time.time() - t0:.0f}s")

        paths = []
        for h in test_hashes:
            paths += list((out / "images/test").glob(f"{h}.*"))
        ids = [image_id(imgs[h].filename) for h in test_hashes]
        names = list(CLASSES)
        best = YOLO(str(STATE / "runs" / f"{iv['name']}_{lv['name']}" / "weights/best.pt"))
        dets = []
        if coco_ann:
            for p, r in zip(paths, best.predict([str(p) for p in paths], imgsz=args.imgsz, conf=0.001, max_det=100, verbose=False, stream=True)):
                iid = image_id(imgs[p.stem].filename)
                for (x1, y1, x2, y2), c, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    dets.append({"image_id": iid, "category_id": CLASSES[names[int(c)]], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc})
        m = YOLO(str(STATE / "runs" / f"{iv['name']}_{lv['name']}" / "weights/best.pt")).val(
            data=str(out / "data.yaml"), split="test", imgsz=args.imgsz, plots=False, verbose=False
        )
        print(f"--- held-out test split ({len(ids)} images), scored against SAM3 labels (agreement with teacher) ---")
        print(f"YOLO26n  mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}")
        print("per-class mAP50:", {n: round(float(v), 3) for n, v in zip(names, m.box.ap50)})
        if not coco_ann:
            raise SystemExit(0)
        teacher = json.loads((STATE / "teacher_dets.json").read_text())
        tdets = [d for d in teacher if d["image_id"] in set(ids)]
        print(f"--- held-out test images: {len(ids)} (never trained on), scored against COCO ground truth ---")
        print("YOLO student (trained on SAM3 labels):", coco_eval(dets, ids))
        print("SAM3 teacher, all proposals          :", coco_eval(tdets, ids))
        print(f"SAM3 teacher, score>={args.thr}           :", coco_eval([d for d in tdets if d["score"] >= args.thr], ids))

    elif args.stage == "cross":
        from ultralytics import YOLO

        exports = sorted((STATE / "export").iterdir(), key=lambda p: p.stat().st_mtime)
        out = STATE / "export" / args.export if args.export else exports[-1]
        print("test split of", out.name)
        for run in sorted((STATE / "runs").iterdir()):
            m = YOLO(str(run / "weights/best.pt")).val(data=str(out / "data.yaml"), split="test", imgsz=args.imgsz, plots=False, verbose=False)
            print(f"  model {run.name}: mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} P={m.box.mp:.3f} R={m.box.mr:.3f}", {n: round(float(v), 3) for n, v in zip(CLASSES, m.box.ap50)})
