import contextvars
import io
import json
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import Session, select

from . import autolabel as al
from . import config, db, inference, pipeline, store, training, versions
from .db import Annotation, Image, ImageVersion, LabelVersion, ModelVersion, WsClass


class AnnotationIn(BaseModel):
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    source: str = "human"
    status: str = "accepted"
    confidence: float | None = None


class ImageVersionIn(BaseModel):
    note: str = ""
    seed: int = 0
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)


class AutolabelIn(BaseModel):
    classes: list[str] | None = None
    hashes: list[str] | None = None
    min_confidence: float | None = None


class AcceptIn(BaseModel):
    min_confidence: float = 0.5
    hashes: list[str] | None = None
    mark_reviewed: bool = True


class TrainIn(BaseModel):
    image_version: str
    label_version: str | None = None
    include_unopened: bool = False
    base: str = "yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    note: str = ""
    parent: str | None = None


class AutoIn(BaseModel):
    base: str = "yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    relabel_all: bool = False
    seed: int = 0
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    note: str = ""


class LabelVersionIn(BaseModel):
    note: str = ""
    include_unopened: bool = False


def new_shared(detector=None) -> dict:
    """State shared by every dataset: one SAM3 model, one GPU lock, the job table."""
    return {"holder": {"detector": detector}, "lock": threading.Lock(), "jobs": {}, "auto_lock": threading.Lock()}


def create_app(engine=None, detector=None, launcher=None, shared=None, probe=None) -> FastAPI:
    """One dataset's API. `probe()` returns (dataset, model) of a training run anywhere, so the GPU is shared."""
    engine = engine or db.make_engine()
    shared = shared or new_shared(detector)
    holder = shared["holder"]
    app = FastAPI(title="LabelHub")

    def get_session():
        with Session(engine) as s:
            yield s

    def version_or_404(s: Session, table, name: str):
        v = s.get(table, name)
        if not v:
            raise HTTPException(404, f"version {name} not found")
        return v

    @app.get("/classes")
    def get_classes(s: Session = Depends(get_session)):
        return [c.name for c in s.exec(select(WsClass).order_by(WsClass.position)).all()]

    @app.put("/classes")
    def put_classes(names: list[str], s: Session = Depends(get_session)):
        if len(set(names)) != len(names):
            raise HTTPException(422, "duplicate class names")
        for c in s.exec(select(WsClass)).all():
            s.delete(c)
        s.add_all(WsClass(name=n, position=i) for i, n in enumerate(names))
        s.commit()
        return names

    @app.post("/images")
    async def upload_images(files: list[UploadFile] = File(...), s: Session = Depends(get_session)):
        out = []
        for f in files:
            try:
                out.append(store.add_image(s, await f.read(), f.filename or "image").model_dump())
            except Exception:
                raise HTTPException(422, f"{f.filename} is not a valid image")
        return out

    @app.get("/images")
    def list_images(s: Session = Depends(get_session)):
        return s.exec(select(Image).where(Image.removed == False)).all()  # noqa: E712

    @app.get("/images/{hash}/file")
    def image_file(hash: str, s: Session = Depends(get_session)):
        img = s.get(Image, hash)
        if not img:
            raise HTTPException(404)
        return FileResponse(store.image_path(img.hash, img.ext))

    @app.delete("/images/{hash}")
    def remove_image(hash: str, s: Session = Depends(get_session)):
        img = s.get(Image, hash)
        if not img:
            raise HTTPException(404)
        img.removed = True
        s.add(img)
        s.commit()

    @app.get("/images/{hash}/annotations")
    def list_annotations(hash: str, s: Session = Depends(get_session)):
        return s.exec(select(Annotation).where(Annotation.image_hash == hash)).all()

    @app.post("/images/{hash}/annotations")
    def add_annotation(hash: str, body: AnnotationIn, s: Session = Depends(get_session)):
        if not s.get(Image, hash):
            raise HTTPException(404)
        ann = Annotation(image_hash=hash, **body.model_dump())
        s.add(ann)
        s.commit()
        s.refresh(ann)
        return ann

    @app.put("/annotations/{id}")
    def update_annotation(id: int, body: AnnotationIn, s: Session = Depends(get_session)):
        ann = s.get(Annotation, id)
        if not ann:
            raise HTTPException(404)
        for k, v in body.model_dump().items():
            setattr(ann, k, v)
        s.add(ann)
        s.commit()
        s.refresh(ann)
        return ann

    @app.delete("/annotations/{id}")
    def delete_annotation(id: int, s: Session = Depends(get_session)):
        ann = s.get(Annotation, id)
        if not ann:
            raise HTTPException(404)
        s.delete(ann)
        s.commit()

    @app.post("/images/{hash}/view")
    def view(hash: str, min_confidence: float = 0.5, s: Session = Depends(get_session)):
        if not s.get(Image, hash):
            raise HTTPException(404)
        img = al.view_image(s, hash, min_confidence)
        return {"hash": hash, "reviewed": img.reviewed}

    @app.put("/images/{hash}/gold")
    def set_gold(hash: str, gold: bool = True, s: Session = Depends(get_session)):
        img = s.get(Image, hash)
        if not img:
            raise HTTPException(404)
        if gold and not img.reviewed:
            raise HTTPException(409, "open and check the image before marking it gold: gold labels must be human-checked")
        img.gold = gold
        s.add(img)
        s.commit()
        return {"hash": hash, "gold": gold}

    @app.put("/images/{hash}/reviewed")
    def set_reviewed(hash: str, reviewed: bool = True, s: Session = Depends(get_session)):
        img = s.get(Image, hash)
        if not img:
            raise HTTPException(404)
        img.reviewed = reviewed
        s.add(img)
        s.commit()
        return {"hash": hash, "reviewed": reviewed}

    def unreviewed(s: Session) -> list[str]:
        return [i.hash for i in s.exec(select(Image).where(Image.removed == False, Image.reviewed == False)).all()]  # noqa: E712

    def not_autolabeled(s: Session) -> list[str]:
        return [i.hash for i in s.exec(select(Image).where(Image.removed == False, Image.auto_done == False)).all()]  # noqa: E712

    def training_anywhere(s: Session):
        if probe:
            return probe()
        m = training.running(s)
        return (None, m.name) if m else None

    def gpu_guard(s: Session) -> None:
        if training_anywhere(s):
            raise HTTPException(409, "GPU busy: a training job is running")
        if shared["auto_lock"].locked():
            raise HTTPException(409, "GPU busy: the auto pipeline is running")

    @app.post("/autolabel")
    def run_autolabel(body: AutolabelIn, s: Session = Depends(get_session)):
        gpu_guard(s)
        classes = body.classes or [c.name for c in s.exec(select(WsClass).order_by(WsClass.position)).all()]
        if not classes:
            raise HTTPException(409, "no classes defined")
        if holder["detector"] is None:
            holder["detector"] = al.Sam3Detector()
        hashes = body.hashes if body.hashes is not None else not_autolabeled(s)
        return {"images": len(hashes), "boxes": al.autolabel(s, holder["detector"], hashes, classes, min_confidence=body.min_confidence)}

    @app.post("/proposals/accept")
    def accept(body: AcceptIn, s: Session = Depends(get_session)):
        hashes = body.hashes
        if hashes is None:
            hashes = sorted(set(s.exec(select(Annotation.image_hash).where(Annotation.source.startswith("auto:"))).all()))
        return al.accept_proposals(s, hashes, body.min_confidence, body.mark_reviewed)

    @app.get("/overview")
    def overview(s: Session = Depends(get_session)):
        counts: dict[str, dict[str, int]] = {}
        for h, st, n in s.exec(select(Annotation.image_hash, Annotation.status, func.count()).group_by(Annotation.image_hash, Annotation.status)).all():
            counts.setdefault(h, {})[st] = n
        auto = {h: n for h, n in s.exec(select(Annotation.image_hash, func.count()).where(Annotation.source.startswith("auto:")).group_by(Annotation.image_hash)).all()}
        out = []
        for i in s.exec(select(Image).where(Image.removed == False)).all():  # noqa: E712
            c = counts.get(i.hash, {})
            out.append(
                {
                    "hash": i.hash,
                    "filename": i.filename,
                    "width": i.width,
                    "height": i.height,
                    "reviewed": i.reviewed,
                    "accepted": c.get("accepted", 0),
                    "proposed": c.get("proposed", 0),
                    "rejected": c.get("rejected", 0),
                    "auto": auto.get(i.hash, 0),
                    "auto_done": i.auto_done,
                    "gold": i.gold,
                }
            )
        return out

    @app.get("/images/{hash}/thumb")
    def image_thumb(hash: str, size: int = 256, s: Session = Depends(get_session)):
        img = s.get(Image, hash)
        if not img:
            raise HTTPException(404)
        size = max(32, min(size, 1024))
        path = config.data_dir() / "thumbs" / f"{hash}_{size}.jpg"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with PILImage.open(store.image_path(img.hash, img.ext)) as pil:
                pil = pil.convert("RGB")
                pil.thumbnail((size, size))
                pil.save(path, "JPEG", quality=82)
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    jobs: dict[str, dict] = shared["jobs"]
    job_lock: threading.Lock = shared["lock"]

    @app.post("/jobs/autolabel")
    def start_autolabel(body: AutolabelIn, s: Session = Depends(get_session)):
        gpu_guard(s)
        classes = body.classes or [c.name for c in s.exec(select(WsClass).order_by(WsClass.position)).all()]
        if not classes:
            raise HTTPException(409, "no classes defined")
        if not job_lock.acquire(blocking=False):
            raise HTTPException(409, "an auto-label job is already running")
        hashes = body.hashes if body.hashes is not None else not_autolabeled(s)
        job = {"id": uuid.uuid4().hex[:8], "status": "running", "done": 0, "total": len(hashes), "boxes": 0, "error": None}
        jobs[job["id"]] = job

        def run():
            try:
                if holder["detector"] is None:
                    job["status"] = "loading model"
                    holder["detector"] = al.Sam3Detector()
                    job["status"] = "running"
                for h in hashes:
                    with Session(engine) as ws:
                        job["boxes"] += al.autolabel(ws, holder["detector"], [h], classes, min_confidence=body.min_confidence)
                    job["done"] += 1
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                job["status"], job["error"] = "failed", str(e)
            finally:
                job_lock.release()

        threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True).start()
        return job

    @app.get("/jobs/{id}")
    def get_job(id: str):
        if id not in jobs:
            raise HTTPException(404)
        return jobs[id]

    @app.get("/status")
    def status(s: Session = Depends(get_session)):
        run = training_anywhere(s)
        busy = "training" if run else ("autolabel" if job_lock.locked() else ("auto" if shared["auto_lock"].locked() else None))
        return {"gpu_busy": busy, "training": run[1] if run else None, "training_dataset": run[0] if run else None}

    auto_runs: dict[str, dict] = {}
    auto_cancel: dict[str, threading.Event] = {}

    @app.post("/auto/run")
    def auto_run(body: AutoIn, s: Session = Depends(get_session)):
        if body.base not in training.BASES:
            raise HTTPException(422, f"base must be one of {training.BASES}")
        if not (1 <= body.epochs <= 500 and 160 <= body.imgsz <= 1280 and 1 <= body.batch <= 64):
            raise HTTPException(422, "epochs 1-500, imgsz 160-1280, batch 1-64")
        if not s.exec(select(WsClass)).first():
            raise HTTPException(409, "define your classes first")
        if not s.exec(select(Image).where(Image.removed == False)).first():  # noqa: E712
            raise HTTPException(409, "add images first")
        gpu_guard(s)
        if job_lock.locked() or (probe and probe()):
            raise HTTPException(409, "GPU busy")
        if not shared["auto_lock"].acquire(blocking=False):
            raise HTTPException(409, "an auto pipeline is already running")
        state = pipeline.new_state(uuid.uuid4().hex[:8], body.model_dump())
        cancelled = threading.Event()
        auto_runs[state["id"]], auto_cancel[state["id"]] = state, cancelled

        def go():
            try:
                pipeline.run(engine, shared, body.model_dump(), state, launcher or training.default_launcher, cancelled)
            finally:
                shared["auto_lock"].release()

        threading.Thread(target=contextvars.copy_context().run, args=(go,), daemon=True).start()
        return state

    @app.get("/auto")
    def auto_latest():
        return list(auto_runs.values())[-1] if auto_runs else None

    @app.post("/auto/{id}/cancel")
    def auto_stop(id: str):
        if id not in auto_runs:
            raise HTTPException(404)
        auto_cancel[id].set()
        return auto_runs[id]

    @app.get("/train-preview")
    def train_preview(image_version: str, label_version: str | None = None, include_unopened: bool = False, s: Session = Depends(get_session)):
        iv = version_or_404(s, ImageVersion, image_version)
        lv = version_or_404(s, LabelVersion, label_version) if label_version else None
        try:
            return {**training.preview(s, iv, lv, include_unopened), "label_version": label_version, "snapshot": lv is None}
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/models")
    def list_models(s: Session = Depends(get_session)):
        training.refresh(s)
        models = s.exec(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()
        st = inference.stats(s, [m.name for m in models])
        return [{**training.summary(m), "inference": st.get(m.name)} for m in models]

    @app.post("/models/{name}/predict")
    def predict(
        name: str,
        file: UploadFile | None = File(None),
        hash: str | None = Form(None),
        conf: float = Form(0.05),
        imgsz: int | None = Form(None),
        s: Session = Depends(get_session),
    ):
        m = version_or_404(s, ModelVersion, name)
        weights = training.model_dir(name) / "best.pt"
        if m.status != "done" or not weights.exists():
            raise HTTPException(409, f"model {name} has no weights yet")
        if not (0 < conf < 1):
            raise HTTPException(422, "conf must be between 0 and 1")
        if file is not None:
            try:
                img = PILImage.open(io.BytesIO(file.file.read())).convert("RGB")
            except Exception:
                raise HTTPException(422, "that file is not a valid image")
            source = "api"
        elif hash:
            row = s.get(Image, hash)
            if not row:
                raise HTTPException(404, "image not found")
            img = PILImage.open(store.image_path(row.hash, row.ext)).convert("RGB")
            source = "ui"
        else:
            raise HTTPException(422, "send an image file or the hash of a dataset image")
        busy = bool(training_anywhere(s)) or shared["auto_lock"].locked()
        if imgsz is not None and not (160 <= imgsz <= 1280):
            raise HTTPException(422, "imgsz must be between 160 and 1280")
        imgsz = imgsz or int(json.loads(m.config_json or "{}").get("imgsz") or 640)
        return inference.predict(s, name, weights, img, conf, imgsz, inference.pick_device(busy), source)

    @app.get("/models/{name}/inference")
    def inference_stats(name: str, s: Session = Depends(get_session)):
        version_or_404(s, ModelVersion, name)
        return inference.stats(s, [name]).get(name)

    @app.post("/models/train")
    def train(body: TrainIn, s: Session = Depends(get_session)):
        if body.base not in training.BASES:
            raise HTTPException(422, f"base must be one of {training.BASES}")
        if not (1 <= body.epochs <= 500 and 160 <= body.imgsz <= 1280 and 1 <= body.batch <= 64):
            raise HTTPException(422, "epochs 1-500, imgsz 160-1280, batch 1-64")
        if job_lock.locked():
            raise HTTPException(409, "GPU busy: an auto-label job is running")
        if shared["auto_lock"].locked():
            raise HTTPException(409, "GPU busy: the auto pipeline is running")
        if probe and probe():
            raise HTTPException(409, "GPU busy: a training job is already running in another dataset")
        pipeline.close_detector(holder)  # SAM3 would hold ~2 GB of a small GPU
        try:
            m = training.start(
                s, body.image_version, body.label_version, body.base, body.epochs, body.imgsz, body.batch,
                body.note, body.parent, launcher or training.default_launcher, include_unopened=body.include_unopened,
            )
        except ValueError as e:
            raise HTTPException(409, str(e))
        return training.summary(m)

    @app.get("/models/{name}")
    def get_model(name: str, s: Session = Depends(get_session)):
        training.refresh(s)
        return training.summary(version_or_404(s, ModelVersion, name))

    @app.post("/models/{name}/cancel")
    def cancel_model(name: str, s: Session = Depends(get_session)):
        try:
            return training.summary(training.cancel(s, name))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/models/{name}/log")
    def model_log(name: str, tail: int = 80, s: Session = Depends(get_session)):
        version_or_404(s, ModelVersion, name)
        path = training.model_dir(name) / "train.log"
        if not path.exists():
            return {"log": ""}
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", path.read_text(errors="replace").replace("\r", "\n"))
        lines = [ln for ln in text.splitlines() if ln.strip() and "━" not in ln]
        return {"log": "\n".join(lines[-tail:])}

    @app.get("/models/{name}/weights")
    def model_weights(name: str, s: Session = Depends(get_session)):
        version_or_404(s, ModelVersion, name)
        path = training.model_dir(name) / "best.pt"
        if not path.exists():
            raise HTTPException(404, "no weights for this model")
        return FileResponse(path, filename=f"{name}.pt")

    @app.post("/models/import")
    async def import_model(
        weights: UploadFile = File(...),
        image_version: str = Form(...),
        label_version: str = Form(...),
        note: str = Form(""),
        map50: float | None = Form(None),
        s: Session = Depends(get_session),
    ):
        version_or_404(s, ImageVersion, image_version)
        version_or_404(s, LabelVersion, label_version)
        m = ModelVersion(
            name=versions.next_name(s, ModelVersion), note=note, image_version=image_version, label_version=label_version,
            base="imported", status="done", imported=True,
            metrics_json=json.dumps({"map50": map50} if map50 is not None else {}),
        )
        d = training.model_dir(m.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "best.pt").write_bytes(await weights.read())
        s.add(m)
        s.commit()
        s.refresh(m)
        return training.summary(m)

    @app.post("/image-versions")
    def make_image_version(body: ImageVersionIn, s: Session = Depends(get_session)):
        try:
            return image_summary(versions.create_image_version(s, body.note, body.seed, body.ratios))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/image-versions")
    def list_image_versions(s: Session = Depends(get_session)):
        return [image_summary(v) for v in s.exec(select(ImageVersion).order_by(ImageVersion.created_at)).all()]

    @app.get("/image-versions/{a}/diff/{b}")
    def image_diff(a: str, b: str, s: Session = Depends(get_session)):
        return versions.diff_images(version_or_404(s, ImageVersion, a), version_or_404(s, ImageVersion, b))

    @app.post("/label-versions")
    def make_label_version(body: LabelVersionIn, s: Session = Depends(get_session)):
        try:
            return label_summary(versions.create_label_version(s, body.note, include_unopened=body.include_unopened))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/label-versions")
    def list_label_versions(s: Session = Depends(get_session)):
        return [label_summary(v) for v in s.exec(select(LabelVersion).order_by(LabelVersion.created_at)).all()]

    @app.get("/label-versions/{a}/diff/{b}")
    def label_diff(a: str, b: str, s: Session = Depends(get_session)):
        return versions.diff_labels(version_or_404(s, LabelVersion, a), version_or_404(s, LabelVersion, b))

    @app.get("/export")
    def export(image_version: str, label_version: str | None = None, bundle: bool = False, s: Session = Depends(get_session)):
        iv = version_or_404(s, ImageVersion, image_version)
        try:
            lv = (
                version_or_404(s, LabelVersion, label_version)
                if label_version
                else versions.create_label_version(s, "auto snapshot for export", reuse_if_same=True)
            )
            with tempfile.TemporaryDirectory() as tmp:
                name = f"{iv.name}_{lv.name}"
                out = versions.export_yolo(s, iv, lv, Path(tmp) / name)
                if bundle:
                    (out / "train.py").write_text(TRAIN_SCRIPT)
                shutil.make_archive(str(Path(tmp) / "zip"), "zip", out)
                buf = io.BytesIO((Path(tmp) / "zip.zip").read_bytes())
        except ValueError as e:
            raise HTTPException(409, str(e))
        return StreamingResponse(
            buf, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}.zip"'}
        )

    return app


TRAIN_SCRIPT = '''# Train on Colab / Kaggle / any GPU box, then import best.pt back into LabelHub (Models > Import).
#   pip install ultralytics
#   python train.py
from pathlib import Path
from ultralytics import YOLO

data = Path(__file__).parent / "data.yaml"
model = YOLO("yolo26n.pt")
model.train(data=str(data), epochs=50, imgsz=640, batch=16, project="runs", name="labelhub", exist_ok=True)
print("weights:", Path("runs/labelhub/weights/best.pt").resolve())
'''


def image_summary(v: ImageVersion) -> dict:
    split = json.loads(v.split_json)
    counts = {"train": 0, "val": 0, "test": 0, "gold": 0}
    for sp in split["assign"].values():
        counts[sp] += 1
    return {
        "name": v.name,
        "parent": v.parent,
        "created_at": v.created_at,
        "note": v.note,
        "images": len(split["assign"]),
        "split": counts,
        "hash": v.hash,
    }


def label_summary(v: LabelVersion) -> dict:
    labels = json.loads(v.labels_json)
    return {
        "name": v.name,
        "parent": v.parent,
        "created_at": v.created_at,
        "note": v.note,
        "classes": json.loads(v.classes_json),
        "images": len(labels),
        "boxes": sum(len(b) for b in labels.values()),
        "sources": json.loads(v.stats_json or "{}"),
        "hash": v.hash,
    }
