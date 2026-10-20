"""Training worker: `python -m labelhub.train_worker <model name>`. Runs in its own process."""
import json
import shutil
import sys
import time
import traceback

from sqlmodel import Session

from . import config, db, training
from .db import ImageVersion, LabelVersion, ModelVersion


def _update(engine, name: str, **fields) -> None:
    with Session(engine) as s:
        m = s.get(ModelVersion, name)
        if m.status == "cancelled":
            raise SystemExit(0)
        for k, v in fields.items():
            setattr(m, k, v)
        s.add(m)
        s.commit()


def run(name: str) -> None:
    from ultralytics import YOLO

    engine = db.make_engine()
    with Session(engine) as s:
        m = s.get(ModelVersion, name)
        cfg = json.loads(m.config_json)
        iv, lv = s.get(ImageVersion, m.image_version), s.get(LabelVersion, m.label_version)
        out = training.model_dir(name)
        classes = json.loads(lv.classes_json)
        yaml = training.prepare_dataset(s, iv, lv, out / "dataset")
    print(f"dataset ready: {yaml}", flush=True)

    weights_dir = config.DATA_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    base = cfg["base"]
    model = YOLO(base if "/" in base else str(weights_dir / base))

    def on_epoch(trainer):
        live = {k.split("/")[-1].replace("(B)", ""): round(float(v), 4) for k, v in trainer.metrics.items() if k.startswith("metrics/")}
        _update(engine, name, epoch=min(trainer.epoch + 1, trainer.epochs), metrics_json=json.dumps({"live": live}))

    model.add_callback("on_fit_epoch_end", on_epoch)
    t0 = time.time()
    model.train(
        data=str(yaml), epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"], workers=4,
        project=str(out), name="run", exist_ok=True, plots=False,
    )
    best = out / "run" / "weights" / "best.pt"
    shutil.copyfile(best, out / "best.pt")
    res = YOLO(str(best)).val(data=str(yaml), split="test", imgsz=cfg["imgsz"], plots=False, verbose=False)
    metrics = {
        "map50": round(float(res.box.map50), 4),
        "map50_95": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
        "per_class_map50": {c: round(float(v), 4) for c, v in zip(classes, res.box.ap50)},
        "train_seconds": round(time.time() - t0),
        "evaluated_on": "test split of the image version, scored against the label version",
    }
    gold_yaml = out / "dataset" / "gold.yaml"
    if gold_yaml.exists():
        g = YOLO(str(best)).val(data=str(gold_yaml), split="val", imgsz=cfg["imgsz"], plots=False, verbose=False)
        metrics["gold"] = {
            "map50": round(float(g.box.map50), 4),
            "map50_95": round(float(g.box.map), 4),
            "precision": round(float(g.box.mp), 4),
            "recall": round(float(g.box.mr), 4),
            "images": len(list((out / "dataset" / "images" / "gold").iterdir())),
        }
    _update(engine, name, status="done", metrics_json=json.dumps(metrics), epoch=cfg["epochs"])
    print("done", metrics, flush=True)


if __name__ == "__main__":
    n = sys.argv[1]
    try:
        run(n)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _update(db.make_engine(), n, status="failed", error=f"{type(e).__name__}: {e}")
        sys.exit(1)
