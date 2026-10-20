import io
import subprocess
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlmodel import Session

from labelhub import api, config, db, store, training, versions
from labelhub.db import Annotation, WsClass


def png(color):
    buf = io.BytesIO()
    PILImage.new("RGB", (20, 10), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    engine = db.make_engine(f"sqlite:///{tmp_path / 't.db'}")
    procs = []

    def launcher(name):
        p = subprocess.Popen(["sleep", "30"], start_new_session=True)
        procs.append(p)
        return p.pid

    client = TestClient(api.create_app(engine, launcher=launcher))
    client.put("/classes", json=["car"])
    for c in ("red", "blue", "green", "white"):
        h = client.post("/images", files=[("files", (f"{c}.png", png(c), "image/png"))]).json()[0]["hash"]
        client.post(f"/images/{h}/annotations", json={"class_name": "car", "cx": .5, "cy": .5, "w": .3, "h": .3, "source": "auto:sam3", "confidence": .9})
        client.put(f"/images/{h}/reviewed")
    yield client, engine, procs
    for p in procs:
        p.kill()
        p.wait()


def test_label_version_records_sources(env):
    client, _, _ = env
    lv = client.post("/label-versions", json={}).json()
    assert lv["sources"] == {"auto:sam3": 4}


def test_preview_and_train_lifecycle(env):
    client, _, procs = env
    iv = client.post("/image-versions", json={"ratios": [1, 0, 0]}).json()
    pv = client.get("/train-preview", params={"image_version": iv["name"]}).json()
    assert pv["images"] == 4 and pv["split"]["train"] == 4 and pv["snapshot"] is True and pv["label_sources"] == {"auto:sam3": 4}
    assert client.get("/label-versions").json() == []  # previewing must not create a version

    assert client.post("/models/train", json={"image_version": iv["name"], "base": "bogus.pt"}).status_code == 422
    r = client.post("/models/train", json={"image_version": iv["name"], "epochs": 3, "note": "t"})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["status"] == "running" and m["label_version"] and m["image_version"] == iv["name"] and m["name"].endswith(".1")

    assert client.post("/models/train", json={"image_version": iv["name"]}).status_code == 409  # already running
    assert client.post("/autolabel", json={}).status_code == 409  # GPU guard
    assert client.get("/status").json() == {"gpu_busy": "training", "training": m["name"], "training_dataset": None}

    c = client.post(f"/models/{m['name']}/cancel").json()
    assert c["status"] == "cancelled"
    assert client.get("/status").json()["gpu_busy"] is None
    assert client.post(f"/models/{m['name']}/cancel").status_code == 409


def test_dead_worker_marked_failed(env):
    client, _, procs = env
    iv = client.post("/image-versions", json={}).json()
    m = client.post("/models/train", json={"image_version": iv["name"]}).json()
    procs[0].kill()
    procs[0].wait()
    time.sleep(0.1)
    got = client.get(f"/models/{m['name']}").json()
    assert got["status"] == "failed" and "stopped" in got["error"]


def test_import_and_download(env):
    client, _, _ = env
    iv = client.post("/image-versions", json={}).json()
    lv = client.post("/label-versions", json={}).json()
    r = client.post(
        "/models/import",
        data={"image_version": iv["name"], "label_version": lv["name"], "map50": "0.61", "note": "colab"},
        files={"weights": ("best.pt", b"fake-weights", "application/octet-stream")},
    )
    m = r.json()
    assert m["imported"] and m["status"] == "done" and m["metrics"]["map50"] == 0.61 and m["has_weights"]
    assert client.get(f"/models/{m['name']}/weights").content == b"fake-weights"
    assert client.get("/models").json()[0]["name"] == m["name"]


def test_prepare_dataset_falls_back_when_split_empty(env, tmp_path):
    _, engine, _ = env
    with Session(engine) as s:
        iv = versions.create_image_version(s, ratios=(1, 0, 0))
        lv = versions.create_label_version(s)
        yaml = training.prepare_dataset(s, iv, lv, tmp_path / "ds")
    text = yaml.read_text()
    assert "val: images/train" in text and "test: images/train" in text and "0: car" in text


def test_export_bundle_has_train_script(env):
    import zipfile

    client, _, _ = env
    iv = client.post("/image-versions", json={}).json()
    r = client.get("/export", params={"image_version": iv["name"], "bundle": True})
    assert any(n.endswith("train.py") for n in zipfile.ZipFile(io.BytesIO(r.content)).namelist())


def test_migration_adds_stats_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE labelversion (name VARCHAR PRIMARY KEY, parent VARCHAR, created_at DATETIME, note VARCHAR, hash VARCHAR, classes_json VARCHAR, labels_json VARCHAR)")
    con.commit()
    con.close()
    engine = db.make_engine(f"sqlite:///{path}")
    with engine.connect() as c:
        assert "stats_json" in {r[1] for r in c.exec_driver_sql("PRAGMA table_info(labelversion)")}


def test_log_is_cleaned(env, tmp_path):
    client, _, _ = env
    iv = client.post("/image-versions", json={}).json()
    m = client.post("/models/train", json={"image_version": iv["name"]}).json()
    (training.model_dir(m["name"])).mkdir(parents=True, exist_ok=True)
    (training.model_dir(m["name"]) / "train.log").write_text("epoch 1\n\x1b[K  Class 50% ━━━━ 1/2\n\n  all 10 20\n")
    assert client.get(f"/models/{m['name']}/log").json()["log"] == "epoch 1\n  all 10 20"


def test_migration_marks_already_labelled_images(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE image (hash VARCHAR PRIMARY KEY, filename VARCHAR, ext VARCHAR, width INT, height INT, removed BOOLEAN, reviewed BOOLEAN)")
    con.execute("CREATE TABLE annotation (id INTEGER PRIMARY KEY, image_hash VARCHAR, class_name VARCHAR, cx FLOAT, cy FLOAT, w FLOAT, h FLOAT, source VARCHAR, status VARCHAR, confidence FLOAT)")
    con.execute("INSERT INTO image VALUES ('a','a.jpg','jpg',1,1,0,0), ('b','b.jpg','jpg',1,1,0,0)")
    con.execute("INSERT INTO annotation VALUES (1,'a','car',.5,.5,.1,.1,'auto:sam3','proposed',.9)")
    con.commit()
    con.close()
    engine = db.make_engine(f"sqlite:///{path}")
    with engine.connect() as c:
        rows = dict(c.exec_driver_sql("SELECT hash, auto_done FROM image").fetchall())
    assert rows == {"a": 1, "b": 0}


def test_refresh_does_not_fail_a_run_that_is_still_launching(env):
    from datetime import datetime, timedelta, timezone

    from sqlmodel import Session

    from labelhub.db import ModelVersion

    _, engine, _ = env
    with Session(engine) as s:
        s.add(ModelVersion(name="2026.10.7", image_version="x", label_version="y", base="yolo26n.pt", status="running", pid=None))
        s.add(ModelVersion(name="2026.10.8", image_version="x", label_version="y", base="yolo26n.pt", status="running", pid=None,
                           created_at=datetime.now(timezone.utc) - timedelta(minutes=5)))
        s.commit()
        training.refresh(s)
        assert s.get(ModelVersion, "2026.10.7").status == "running"  # pid not recorded yet: the worker is being started
        assert s.get(ModelVersion, "2026.10.8").status == "failed"  # never got a worker: really failed
