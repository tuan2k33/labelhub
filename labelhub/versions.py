import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, select

from .db import Annotation, Image, ImageVersion, LabelVersion, WsClass
from .store import image_path


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def next_name(session: Session, table, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    prefix = f"{now.year}.{now.month:02d}."
    used = [int(v.name[len(prefix):]) for v in session.exec(select(table)).all() if v.name.startswith(prefix)]
    return f"{prefix}{max(used, default=0) + 1}"


def latest(session: Session, table):
    return session.exec(select(table).order_by(table.created_at.desc())).first()


def split_of(media_hash: str, seed: int, ratios: tuple[float, float, float]) -> str:
    x = int(hashlib.sha256(f"{seed}:{media_hash}".encode()).hexdigest(), 16) / 16**64
    train, val, _ = ratios
    return "train" if x < train else "val" if x < train + val else "test"


def create_image_version(
    session: Session,
    note: str = "",
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> ImageVersion:
    images = session.exec(select(Image).where(Image.removed == False)).all()  # noqa: E712
    media = sorted(i.hash for i in images)
    if not media:
        raise ValueError("no images")
    gold = {i.hash for i in images if i.gold}
    assign = {h: "gold" if h in gold else split_of(h, seed, ratios) for h in media}
    split = {"seed": seed, "ratios": list(ratios), "assign": assign}
    digest = _digest({"media": media, "split": split})
    last = latest(session, ImageVersion)
    if last and last.hash == digest:
        raise ValueError(f"no changes since {last.name}")
    v = ImageVersion(
        name=next_name(session, ImageVersion),
        parent=last.name if last else None,
        note=note,
        hash=digest,
        media_json=json.dumps(media),
        split_json=json.dumps(split),
    )
    session.add(v)
    session.commit()
    session.refresh(v)
    return v


def current_labels(session: Session, include_unopened: bool = False) -> dict:
    """Labels as they would be snapshotted now (accepted boxes on reviewed images), without saving.
    With include_unopened, images SAM3 has labelled but nobody opened yet count too (Auto mode)."""
    classes = [c.name for c in session.exec(select(WsClass).order_by(WsClass.position)).all()]
    if not classes:
        raise ValueError("no classes defined")
    index = {c: i for i, c in enumerate(classes)}
    labels = {}
    stats: dict[str, int] = {}
    chosen = Image.reviewed == True  # noqa: E712
    if include_unopened:
        chosen = (Image.reviewed == True) | (Image.auto_done == True)  # noqa: E712
    for img in session.exec(select(Image).where(Image.removed == False, chosen)).all():  # noqa: E712
        anns = session.exec(
            select(Annotation).where(Annotation.image_hash == img.hash, Annotation.status == "accepted")
        ).all()
        unknown = {a.class_name for a in anns} - set(index)
        if unknown:
            raise ValueError(f"annotations use unknown classes: {sorted(unknown)}")
        for a in anns:
            stats[a.source] = stats.get(a.source, 0) + 1
        labels[img.hash] = sorted(
            [index[a.class_name], round(a.cx, 6), round(a.cy, 6), round(a.w, 6), round(a.h, 6)] for a in anns
        )
    if not labels:
        raise ValueError("no labelled images" if include_unopened else "no reviewed images")
    return {"classes": classes, "labels": labels, "stats": stats, "digest": _digest({"classes": classes, "labels": labels})}


def create_label_version(session: Session, note: str = "", reuse_if_same: bool = False, include_unopened: bool = False) -> LabelVersion:
    cur = current_labels(session, include_unopened)
    last = latest(session, LabelVersion)
    if last and last.hash == cur["digest"]:
        if reuse_if_same:
            return last
        raise ValueError(f"no changes since {last.name}")
    v = LabelVersion(
        name=next_name(session, LabelVersion),
        parent=last.name if last else None,
        note=note,
        hash=cur["digest"],
        classes_json=json.dumps(cur["classes"]),
        labels_json=json.dumps(cur["labels"], sort_keys=True),
        stats_json=json.dumps(cur["stats"], sort_keys=True),
    )
    session.add(v)
    session.commit()
    session.refresh(v)
    return v


def diff_images(a: ImageVersion, b: ImageVersion) -> dict:
    ma, mb = set(json.loads(a.media_json)), set(json.loads(b.media_json))
    return {"images_added": sorted(mb - ma), "images_removed": sorted(ma - mb)}


def diff_labels(a: LabelVersion, b: LabelVersion) -> dict:
    la, lb = json.loads(a.labels_json), json.loads(b.labels_json)
    ca, cb = json.loads(a.classes_json), json.loads(b.classes_json)
    return {
        "images_added": sorted(set(lb) - set(la)),
        "images_removed": sorted(set(la) - set(lb)),
        "labels_changed": sorted(h for h in set(la) & set(lb) if la[h] != lb[h]),
        "classes_added": [c for c in cb if c not in ca],
        "classes_removed": [c for c in ca if c not in cb],
    }


def train_data(iv: ImageVersion, lv: LabelVersion | None = None, labels: dict | None = None) -> dict:
    """Images in the image version that have labels (in the label version, or the given labels)."""
    assign = json.loads(iv.split_json)["assign"]
    labels = labels if labels is not None else json.loads(lv.labels_json)
    return {h: {"split": assign[h], "boxes": labels[h]} for h in assign if h in labels}


def export_yolo(session: Session, iv: ImageVersion, lv: LabelVersion, out_dir: Path) -> Path:
    data = train_data(iv, lv)
    if not data:
        raise ValueError("image version and label version share no images")
    classes = json.loads(lv.classes_json)
    exts = {i.hash: i.ext for i in session.exec(select(Image)).all()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for h, item in data.items():
        s = item["split"]
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path(h, exts[h]), out_dir / "images" / s / f"{h}.{exts[h]}")
        lines = [f"{c} {cx} {cy} {w} {hh}" for c, cx, cy, w, hh in item["boxes"]]
        (out_dir / "labels" / s / f"{h}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    return out_dir
