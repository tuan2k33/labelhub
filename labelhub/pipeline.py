"""Auto mode: label new images with SAM3, snapshot versions, train YOLO, evaluate. No human in the loop."""
import threading
import time

from sqlmodel import Session, select

from . import autolabel as al
from . import training, versions
from .db import Image, ImageVersion, LabelVersion, ModelVersion, WsClass

POLL = 2.0
STEPS = ["label", "versions", "train", "evaluate"]
MIN_CONF = 0.5


class Cancelled(Exception):
    pass


def new_state(pid: str, cfg: dict) -> dict:
    return {
        "id": pid, "status": "running", "step": "label", "cfg": cfg, "error": None, "model": None,
        "label": {"done": 0, "total": 0}, "versions": None, "epoch": 0, "epochs": cfg["epochs"], "detail": "",
        "started": time.time(), "metrics": None,
    }


def close_detector(holder: dict) -> None:
    det = holder.get("detector")
    holder["detector"] = None
    if det is not None and hasattr(det, "close"):
        det.close()


def run(engine, shared: dict, cfg: dict, state: dict, launcher, cancelled: threading.Event, poll: float | None = None) -> None:
    poll = POLL if poll is None else poll
    lock, holder = shared["lock"], shared["holder"]
    try:
        with Session(engine) as s:
            classes = [c.name for c in s.exec(select(WsClass).order_by(WsClass.position)).all()]
            if not classes:
                raise ValueError("define your classes first")
            q = select(Image).where(Image.removed == False)  # noqa: E712
            if not cfg.get("relabel_all"):
                q = q.where(Image.auto_done == False)  # noqa: E712
            todo = [i.hash for i in s.exec(q).all()]
        state["label"]["total"] = len(todo)

        if todo:
            if not lock.acquire(blocking=False):
                raise RuntimeError("GPU busy: an auto-label job is running")
            try:
                if holder["detector"] is None:
                    state["detail"] = "loading SAM3"
                    holder["detector"] = al.Sam3Detector()
                state["detail"] = ""
                for h in todo:
                    if cancelled.is_set():
                        raise Cancelled
                    with Session(engine) as ws:
                        al.autolabel(ws, holder["detector"], [h], classes, min_confidence=MIN_CONF)
                    state["label"]["done"] += 1
            finally:
                close_detector(holder)
                lock.release()

        state["step"] = "versions"
        with Session(engine) as s:
            try:
                iv = versions.create_image_version(s, "auto pipeline", cfg.get("seed", 0), tuple(cfg.get("ratios", (0.8, 0.1, 0.1))))
            except ValueError as e:
                if not str(e).startswith("no changes"):
                    raise
                iv = versions.latest(s, ImageVersion)
            lv = versions.create_label_version(s, "auto pipeline", reuse_if_same=True, include_unopened=True)
            state["versions"] = {"image": iv.name, "label": lv.name}

        if cancelled.is_set():
            raise Cancelled
        state["step"] = "train"
        with Session(engine) as s:
            m = training.start(
                s, iv.name, lv.name, cfg["base"], cfg["epochs"], cfg["imgsz"], cfg["batch"],
                cfg.get("note") or "auto pipeline", None, launcher,
            )
            name = m.name
        state["model"] = name

        while True:
            if cancelled.is_set():
                with Session(engine) as s:
                    training.cancel(s, name)
                raise Cancelled
            with Session(engine) as s:
                training.refresh(s)
                m = s.get(ModelVersion, name)
                state["epoch"], status, err, metrics = m.epoch, m.status, m.error, m.metrics_json
            if status == "running":
                state["step"] = "train" if state["epoch"] < state["epochs"] else "evaluate"
                time.sleep(poll)
                continue
            if status != "done":
                raise RuntimeError(err or f"training {status}")
            break
        import json

        state["metrics"] = json.loads(metrics or "{}")
        state["step"], state["status"] = "done", "done"
    except Cancelled:
        state["status"] = "cancelled"
    except Exception as e:  # noqa: BLE001
        state["status"], state["error"] = "failed", f"{type(e).__name__}: {e}" if not isinstance(e, (ValueError, RuntimeError)) else str(e)
