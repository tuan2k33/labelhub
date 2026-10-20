import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from labelhub import api, config, db, inference

WEIGHTS = Path.home() / ".dev/data/lh_coco/weights/yolo26n.pt"
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="needs a local yolo26n.pt")


def png(color, size=(64, 48)):
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(inference, "DEVICE", "cpu")
    inference._models.clear()
    c = TestClient(api.create_app(db.make_engine(f"sqlite:///{tmp_path / 't.db'}")))
    c.put("/classes", json=["person"])
    h = c.post("/images", files=[("files", ("a.png", png("red"), "image/png"))]).json()[0]["hash"]
    c.post(f"/images/{h}/annotations", json={"class_name": "person", "cx": .5, "cy": .5, "w": .3, "h": .3})
    c.put(f"/images/{h}/reviewed")
    iv = c.post("/image-versions", json={}).json()
    lv = c.post("/label-versions", json={}).json()
    m = c.post(
        "/models/import",
        data={"image_version": iv["name"], "label_version": lv["name"]},
        files={"weights": ("best.pt", WEIGHTS.read_bytes(), "application/octet-stream")},
    ).json()
    yield c, m["name"], h


def test_predict_upload_and_dataset_image(env):
    c, name, h = env
    r = c.post(f"/models/{name}/predict", files={"file": ("x.png", png("blue", (320, 240)), "image/png")}, data={"conf": "0.05"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["model"], body["width"], body["height"], body["device"], body["cold"]) == (name, 320, 240, "cpu", True)
    assert body["total_ms"] > 0 and isinstance(body["detections"], list)
    r2 = c.post(f"/models/{name}/predict", data={"hash": h})
    assert r2.status_code == 200 and r2.json()["cold"] is False and (r2.json()["width"], r2.json()["height"]) == (64, 48)


def test_stats_leave_out_the_cold_run(env):
    c, name, h = env
    assert c.get(f"/models/{name}/inference").json() is None
    for _ in range(4):
        c.post(f"/models/{name}/predict", data={"hash": h})
    st = c.get(f"/models/{name}/inference").json()
    assert (st["runs"], st["warm_runs"], st["device"]) == (4, 3, "cpu")
    assert 0 < st["p50_ms"] <= st["p95_ms"]
    listed = [m for m in c.get("/models").json() if m["name"] == name][0]
    assert listed["inference"]["runs"] == 4


def test_predict_errors(env):
    c, name, h = env
    assert c.post("/models/2000.01.1/predict", data={"hash": h}).status_code == 404
    assert c.post(f"/models/{name}/predict").status_code == 422  # no image
    assert c.post(f"/models/{name}/predict", data={"hash": "nope"}).status_code == 404
    assert c.post(f"/models/{name}/predict", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 422
    assert c.post(f"/models/{name}/predict", data={"hash": h, "conf": "1.5"}).status_code == 422


def test_model_without_weights_is_refused(env, tmp_path):
    c, name, h = env
    from sqlmodel import Session

    from labelhub.db import ModelVersion

    engine = db.make_engine(f"sqlite:///{tmp_path / 't.db'}")
    with Session(engine) as s:
        s.add(ModelVersion(name="2026.10.9", image_version="x", label_version="y", base="yolo26n.pt", status="running"))
        s.commit()
    assert c.post("/models/2026.10.9/predict", data={"hash": h}).status_code == 409


def test_image_size_can_be_overridden(env):
    c, name, h = env
    r = c.post(f"/models/{name}/predict", data={"hash": h, "imgsz": "320"}).json()
    assert r["imgsz"] == 320
    assert c.post(f"/models/{name}/predict", data={"hash": h, "imgsz": "50"}).status_code == 422
    assert c.post(f"/models/{name}/predict", data={"hash": h}).json()["imgsz"] == 640  # imported model: no size recorded
