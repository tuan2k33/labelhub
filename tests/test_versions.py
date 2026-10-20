import io
import json

import pytest
from PIL import Image as PILImage
from sqlmodel import Session

from labelhub import config, db, store, versions
from labelhub.db import Annotation, ImageVersion, LabelVersion, WsClass


@pytest.fixture
def s(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    engine = db.make_engine(f"sqlite:///{tmp_path / 't.db'}")
    with Session(engine) as session:
        session.add_all([WsClass(name="car", position=0), WsClass(name="bag", position=1)])
        session.commit()
        yield session


def png(color):
    buf = io.BytesIO()
    PILImage.new("RGB", (20, 10), color).save(buf, "PNG")
    return buf.getvalue()


def label(s, img, class_name="bag", status="accepted", reviewed=True, **kw):
    s.add(Annotation(image_hash=img.hash, class_name=class_name, cx=.5, cy=.5, w=.2, h=.4, status=status, **kw))
    img.reviewed = reviewed
    s.add(img)
    s.commit()


def test_add_image_dedupes(s):
    a = store.add_image(s, png("red"), "a.png")
    b = store.add_image(s, png("red"), "copy.png")
    assert a.hash == b.hash and (a.width, a.height) == (20, 10)


def test_image_version_names_and_diff(s):
    a = store.add_image(s, png("red"), "a.png")
    v1 = versions.create_image_version(s, seed=7)
    b = store.add_image(s, png("blue"), "b.png")
    v2 = versions.create_image_version(s, seed=7)
    assert v1.name.endswith(".1") and v2.name.endswith(".2") and v2.parent == v1.name
    assert versions.diff_images(v1, v2) == {"images_added": [b.hash], "images_removed": []}
    assign1 = json.loads(v1.split_json)["assign"]
    assign2 = json.loads(v2.split_json)["assign"]
    assert assign1[a.hash] == assign2[a.hash]


def test_image_version_ignores_label_edits(s):
    img = store.add_image(s, png("red"), "a.png")
    versions.create_image_version(s)
    label(s, img)
    with pytest.raises(ValueError, match="no changes"):
        versions.create_image_version(s)


def test_label_version_only_reviewed_and_accepted(s):
    a = store.add_image(s, png("red"), "a.png")
    b = store.add_image(s, png("blue"), "b.png")
    label(s, a)
    label(s, b, status="proposed", source="auto:sam3")
    c = store.add_image(s, png("green"), "c.png")
    label(s, c, reviewed=False)
    v = versions.create_label_version(s)
    labels = json.loads(v.labels_json)
    assert set(labels) == {a.hash, b.hash} and labels[b.hash] == [] and len(labels[a.hash]) == 1


def test_label_version_changes_and_reuse(s):
    a = store.add_image(s, png("red"), "a.png")
    label(s, a)
    v1 = versions.create_label_version(s)
    with pytest.raises(ValueError, match="no changes"):
        versions.create_label_version(s)
    assert versions.create_label_version(s, reuse_if_same=True).name == v1.name
    ann = s.exec(__import__("sqlmodel").select(Annotation)).first()
    ann.class_name = "car"
    s.add(ann)
    s.commit()
    v2 = versions.create_label_version(s)
    assert v2.parent == v1.name and versions.diff_labels(v1, v2)["labels_changed"] == [a.hash]


def test_unknown_class_and_empty_rejected(s):
    a = store.add_image(s, png("red"), "a.png")
    with pytest.raises(ValueError, match="no reviewed"):
        versions.create_label_version(s)
    label(s, a, class_name="dog")
    with pytest.raises(ValueError, match="unknown classes"):
        versions.create_label_version(s)


def test_train_data_is_intersection(s):
    a = store.add_image(s, png("red"), "a.png")
    iv = versions.create_image_version(s, ratios=(1, 0, 0))
    b = store.add_image(s, png("blue"), "b.png")
    label(s, a)
    label(s, b)
    lv = versions.create_label_version(s)
    data = versions.train_data(iv, lv)
    assert set(data) == {a.hash} and data[a.hash]["split"] == "train"


def test_export_yolo(s, tmp_path):
    img = store.add_image(s, png("red"), "a.png")
    label(s, img)
    iv = versions.create_image_version(s, ratios=(1, 0, 0))
    lv = versions.create_label_version(s)
    out = versions.export_yolo(s, iv, lv, tmp_path / "out")
    assert (out / "labels/train" / f"{img.hash}.txt").read_text() == "1 0.5 0.5 0.2 0.4\n"
    assert "1: bag" in (out / "data.yaml").read_text()
