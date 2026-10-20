import json
import os
from datetime import datetime, timezone
import signal
import subprocess
import sys
from pathlib import Path

from sqlmodel import Session, select

from . import config, versions
from .db import ImageVersion, LabelVersion, ModelVersion

BASES = ["yolo26n.pt", "yolo26s.pt"]
STARTUP_GRACE = 60  # seconds a new run may spend before its worker pid is recorded


def model_dir(name: str) -> Path:
    return config.data_dir() / "models" / name


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True


def refresh(session: Session) -> None:
    """Mark runs whose worker process vanished as failed."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for m in session.exec(select(ModelVersion).where(ModelVersion.status == "running", ModelVersion.imported == False)).all():  # noqa: E712
        launching = m.pid is None and (now - m.created_at.replace(tzinfo=None)).total_seconds() < STARTUP_GRACE
        if not launching and not pid_alive(m.pid):
            m.status, m.error = "failed", m.error or "worker process stopped unexpectedly (see log)"
            session.add(m)
    session.commit()


def running(session: Session) -> ModelVersion | None:
    refresh(session)
    return session.exec(select(ModelVersion).where(ModelVersion.status == "running")).first()


def preview(session: Session, iv: ImageVersion, lv: LabelVersion | None, include_unopened: bool = False) -> dict:
    """What a training run would use. With lv=None, previews a snapshot of the current labels."""
    if lv is not None:
        data = versions.train_data(iv, lv)
        sources, classes = json.loads(lv.stats_json or "{}"), json.loads(lv.classes_json)
    else:
        cur = versions.current_labels(session, include_unopened)
        data = versions.train_data(iv, labels=cur["labels"])
        sources, classes = cur["stats"], cur["classes"]
    counts = {"train": 0, "val": 0, "test": 0, "gold": 0}
    boxes = 0
    for item in data.values():
        counts[item["split"]] += 1
        boxes += len(item["boxes"])
    return {"images": len(data), "split": counts, "boxes": boxes, "label_sources": sources, "classes": classes}


def default_launcher(name: str) -> int:
    d = model_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    log = open(d / "train.log", "ab")
    env = {**os.environ, "LABELHUB_DATA": str(config.data_dir()), "PYTHONPATH": str(config.ROOT)}
    p = subprocess.Popen(
        [sys.executable, "-W", "ignore", "-m", "labelhub.train_worker", name],
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(config.ROOT), start_new_session=True,
    )
    return p.pid


def start(
    session: Session,
    image_version: str,
    label_version: str | None,
    base: str,
    epochs: int,
    imgsz: int,
    batch: int,
    note: str = "",
    parent: str | None = None,
    launcher=default_launcher,
    include_unopened: bool = False,
) -> ModelVersion:
    iv = session.get(ImageVersion, image_version)
    if not iv:
        raise ValueError(f"image version {image_version} not found")
    lv = session.get(LabelVersion, label_version) if label_version else versions.create_label_version(session, "auto snapshot for training", reuse_if_same=True, include_unopened=include_unopened)
    if not lv:
        raise ValueError(f"label version {label_version} not found")
    pv = preview(session, iv, lv)
    if pv["split"]["train"] == 0:
        raise ValueError("no training images: the image version and label version share no train images")
    if running(session):
        raise ValueError("a training job is already running")
    start_from = base
    if parent:
        pm = session.get(ModelVersion, parent)
        if not pm or pm.status != "done":
            raise ValueError(f"parent model {parent} is not available")
        start_from = str(model_dir(parent) / "best.pt")
    m = ModelVersion(
        name=versions.next_name(session, ModelVersion),
        parent=parent,
        note=note,
        image_version=iv.name,
        label_version=lv.name,
        base=base if not parent else f"{parent}",
        config_json=json.dumps({"base": start_from, "epochs": epochs, "imgsz": imgsz, "batch": batch}),
        epochs=epochs,
    )
    session.add(m)
    session.commit()
    session.refresh(m)
    m.pid = launcher(m.name)
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def cancel(session: Session, name: str) -> ModelVersion:
    m = session.get(ModelVersion, name)
    if not m:
        raise ValueError(f"model {name} not found")
    if m.status != "running":
        raise ValueError(f"model {name} is {m.status}, not running")
    if m.pid and pid_alive(m.pid):
        try:
            os.killpg(os.getpgid(m.pid), signal.SIGTERM)
        except OSError:
            pass
    m.status = "cancelled"
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def prepare_dataset(session: Session, iv: ImageVersion, lv: LabelVersion, out_dir: Path) -> Path:
    """Export YOLO data and write a data.yaml that still works when a split is empty."""
    versions.export_yolo(session, iv, lv, out_dir)

    def has(split: str) -> bool:
        d = out_dir / "images" / split
        return d.is_dir() and any(d.iterdir())

    train = "train"
    val = "val" if has("val") else train
    test = "test" if has("test") else val
    classes = json.loads(lv.classes_json)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    yaml = out_dir / "data.yaml"
    yaml.write_text(f"path: {out_dir.resolve()}\ntrain: images/{train}\nval: images/{val}\ntest: images/{test}\nnames:\n{names}\n")
    if has("gold"):  # human-checked images, never trained on, scored separately
        (out_dir / "gold.yaml").write_text(f"path: {out_dir.resolve()}\ntrain: images/gold\nval: images/gold\ntest: images/gold\nnames:\n{names}\n")
    return yaml


def summary(m: ModelVersion) -> dict:
    cfg = json.loads(m.config_json or "{}")
    return {
        "name": m.name,
        "parent": m.parent,
        "created_at": m.created_at,
        "note": m.note,
        "image_version": m.image_version,
        "label_version": m.label_version,
        "base": m.base,
        "config": cfg,
        "status": m.status,
        "epoch": m.epoch,
        "epochs": m.epochs,
        "metrics": json.loads(m.metrics_json or "{}"),
        "error": m.error,
        "imported": m.imported,
        "has_weights": (model_dir(m.name) / "best.pt").exists(),
    }
