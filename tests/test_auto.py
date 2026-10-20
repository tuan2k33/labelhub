import io
import json
import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlmodel import Session

from labelhub import api, config, db, pipeline, store, training, versions
from labelhub.autolabel import Detection
from labelhub.db import Annotation, ModelVersion, WsClass


def png(color):
    buf = io.BytesIO()
    PILImage.new("RGB", (20, 10), color).save(buf, "PNG")
    return buf.getvalue()


class Fake:
    closed = 0

    def detect(self, image, classes):
        w, h = image.size
        return [Detection(classes[0], (0, 0, w / 2, h / 2), 0.9), Detection(classes[0], (w / 2, h / 2, w, h), 0.4)]

    def close(self):
        Fake.closed += 1


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(pipeline, "POLL", 0.05)
    engine = db.make_engine(f"sqlite:///{tmp_path / 't.db'}")
    mode = {"finish": True, "delay": 0.3}
    procs = []

    def launcher(name):
        def finish():
            time.sleep(mode["delay"])
            if not mode["finish"]:
                return
            with Session(engine) as s:
                m = s.get(ModelVersion, name)
                if m.status == "running":
                    m.status, m.epoch = "done", m.epochs
                    m.metrics_json = json.dumps({"map50": 0.5, "gold": {"map50": 0.4, "images": 1}})
                    s.add(m)
                    s.commit()

        threading.Thread(target=finish, daemon=True).start()
        p = subprocess.Popen(["sleep", "30"], start_new_session=True)  # a stand-in worker that cancel can kill
        procs.append(p)
        return p.pid

    Fake.closed = 0
    client = TestClient(api.create_app(engine, detector=Fake(), launcher=launcher))
    client.put("/classes", json=["car"])
    for c in ("red", "blue", "green", "white", "black", "gray"):
        client.post("/images", files=[("files", (f"{c}.png", png(c), "image/png"))])
    yield client, engine, mode
    for p in procs:
        p.kill()
        p.wait()


def wait_auto(client, want=("done", "failed", "cancelled")):
    for _ in range(200):
        st = client.get("/auto").json()
        if st and st["status"] in want:
            return st
        time.sleep(0.05)
    raise AssertionError(client.get("/auto").json())


def test_gold_needs_review_and_is_never_trained_on(env, tmp_path):
    client, engine, _ = env
    ov = client.get("/overview").json()
    a, b = ov[0]["hash"], ov[1]["hash"]
    assert client.put(f"/images/{a}/gold").status_code == 409  # not opened yet
    for h in (a, b):
        client.post(f"/images/{h}/annotations", json={"class_name": "car", "cx": .5, "cy": .5, "w": .2, "h": .2})
        client.put(f"/images/{h}/reviewed")
    assert client.put(f"/images/{a}/gold").json() == {"hash": a, "gold": True}
    assert {i["hash"]: i["gold"] for i in client.get("/overview").json()}[a] is True

    iv = client.post("/image-versions", json={"ratios": [1, 0, 0]}).json()
    assert iv["split"]["gold"] == 1 and iv["split"]["train"] == 5
    lv = client.post("/label-versions", json={}).json()
    pv = client.get("/train-preview", params={"image_version": iv["name"], "label_version": lv["name"]}).json()
    assert pv["split"] == {"train": 1, "val": 0, "test": 0, "gold": 1}
    with Session(engine) as s:
        yaml = training.prepare_dataset(s, s.get(db.ImageVersion, iv["name"]), s.get(db.LabelVersion, lv["name"]), tmp_path / "ds")
    assert (tmp_path / "ds" / "gold.yaml").exists() and "images/gold" in (tmp_path / "ds" / "gold.yaml").read_text()
    assert len(list((tmp_path / "ds/images/gold").iterdir())) == 1
    train_files = {p.stem for p in (tmp_path / "ds/images/train").iterdir()}
    assert a not in train_files  # gold image stays out of training


def test_label_version_can_include_unopened_images(env):
    client, _, _ = env
    assert client.post("/label-versions", json={}).status_code == 409  # nothing opened
    client.post("/autolabel", json={"min_confidence": 0.5})
    assert client.post("/label-versions", json={}).status_code == 409  # labelled by SAM3 but still not opened
    lv = client.post("/label-versions", json={"include_unopened": True}).json()
    assert (lv["images"], lv["boxes"], lv["sources"]) == (6, 6, {"auto:sam3": 6})


def test_auto_pipeline_runs_every_step_and_frees_sam3(env):
    client, _, mode = env
    mode["delay"] = 1.5  # keep the run alive long enough to observe the busy state
    r = client.post("/auto/run", json={"epochs": 2, "imgsz": 320, "batch": 4})
    assert r.status_code == 200, r.text
    assert client.post("/auto/run", json={}).status_code == 409  # already running
    assert client.post("/autolabel", json={}).status_code == 409  # GPU guard while the pipeline runs
    assert client.get("/status").json()["gpu_busy"] in ("auto", "training")  # "training" once the chain reaches its train step

    st = wait_auto(client)
    assert st["status"] == "done", st["error"]
    assert st["label"] == {"done": 6, "total": 6}
    assert st["versions"]["image"].endswith(".1") and st["versions"]["label"].endswith(".1")
    assert st["model"] and st["metrics"]["gold"]["map50"] == 0.4
    assert Fake.closed >= 1  # SAM3 was released before training started
    assert client.get("/status").json()["gpu_busy"] is None

    m = client.get(f"/models/{st['model']}").json()
    assert (m["status"], m["image_version"], m["label_version"], m["note"]) == ("done", st["versions"]["image"], st["versions"]["label"], "auto pipeline")
    lv = client.get("/label-versions").json()[0]
    assert lv["images"] == 6 and lv["sources"] == {"auto:sam3": 6}  # no human involved

    # a second run has nothing new to label and reuses the versions
    client.post("/auto/run", json={"epochs": 1})
    st2 = wait_auto(client)
    assert st2["id"] != st["id"] and st2["label"]["total"] == 0 and st2["versions"] == st["versions"]


def test_auto_pipeline_cancel(env):
    client, _, mode = env
    mode["finish"] = False
    st = client.post("/auto/run", json={"epochs": 5}).json()
    for _ in range(100):
        if client.get("/auto").json()["model"]:
            break
        time.sleep(0.05)
    client.post(f"/auto/{st['id']}/cancel")
    final = wait_auto(client)
    assert final["status"] == "cancelled"
    assert client.get(f"/models/{final['model']}").json()["status"] == "cancelled"
    assert client.get("/status").json()["gpu_busy"] is None


def test_auto_requires_classes_and_images(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    c = TestClient(api.create_app(db.make_engine(f"sqlite:///{tmp_path / 't.db'}"), detector=Fake()))
    assert c.post("/auto/run", json={}).status_code == 409  # no classes
    c.put("/classes", json=["car"])
    assert c.post("/auto/run", json={}).status_code == 409  # no images
