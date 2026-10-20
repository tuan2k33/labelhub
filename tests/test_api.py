import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from labelhub import api, config, db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    return TestClient(api.create_app(db.make_engine(f"sqlite:///{tmp_path / 't.db'}")))


def png(color):
    buf = io.BytesIO()
    PILImage.new("RGB", (20, 10), color).save(buf, "PNG")
    return buf.getvalue()


def test_upload_many_returns_all_fields(client):
    r = client.post("/images", files=[("files", (f"{c}.png", png(c), "image/png")) for c in ("red", "blue")])
    assert [i["filename"] for i in r.json()] == ["red.png", "blue.png"]


def test_flow(client):
    assert client.put("/classes", json=["car", "bag"]).status_code == 200
    r = client.post("/images", files=[("files", ("a.png", png("red"), "image/png"))])
    h = r.json()[0]["hash"]
    assert client.post("/images", files=[("files", ("x.txt", b"nope", "text/plain"))]).status_code == 422

    iv = client.post("/image-versions", json={"ratios": [1, 0, 0]}).json()
    assert iv["images"] == 1 and client.post("/image-versions", json={"ratios": [1, 0, 0]}).status_code == 409

    assert client.post("/label-versions", json={}).status_code == 409
    client.post(f"/images/{h}/annotations", json={"class_name": "bag", "cx": .5, "cy": .5, "w": .2, "h": .2})
    client.put(f"/images/{h}/reviewed")
    lv1 = client.post("/label-versions", json={"note": "first"}).json()
    assert (lv1["images"], lv1["boxes"]) == (1, 1)

    ann = client.get(f"/images/{h}/annotations").json()[0]
    client.put(f"/annotations/{ann['id']}", json={**ann, "class_name": "car"})
    lv2 = client.post("/label-versions", json={}).json()
    assert lv2["parent"] == lv1["name"]
    assert client.get(f"/label-versions/{lv1['name']}/diff/{lv2['name']}").json()["labels_changed"] == [h]

    r = client.get("/export", params={"image_version": iv["name"], "label_version": lv1["name"]})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert any(n.endswith("data.yaml") for n in z.namelist()) and any(f"{h}.txt" in n for n in z.namelist())
    assert client.get("/export", params={"image_version": iv["name"]}).status_code == 200
    assert client.get("/export", params={"image_version": "nope"}).status_code == 404
    assert client.get(f"/images/{h}/file").status_code == 200


class FakeDetector:
    def detect(self, image, classes):
        from labelhub.autolabel import Detection

        w, h = image.size
        return [Detection(classes[0], (0, 0, w / 2, h / 2), 0.9), Detection(classes[0], (w / 2, h / 2, w, h), 0.2)]


def test_auto_mode_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    c = TestClient(api.create_app(db.make_engine(f"sqlite:///{tmp_path / 't.db'}"), detector=FakeDetector()))
    c.put("/classes", json=["car"])
    h = c.post("/images", files=[("files", ("a.png", png("red"), "image/png"))]).json()[0]["hash"]
    assert c.post("/autolabel", json={}).json() == {"images": 1, "boxes": 2}
    assert c.post("/autolabel", json={}).json() == {"images": 0, "boxes": 0}  # nothing new to label
    assert c.get("/overview").json()[0]["auto_done"] is True
    assert c.post("/autolabel", json={"hashes": [h]}).json() == {"images": 1, "boxes": 2}  # re-run replaces, no duplicates
    assert len(c.get(f"/images/{h}/annotations").json()) == 2
    assert {a["status"] for a in c.get(f"/images/{h}/annotations").json()} == {"proposed"}
    assert c.post("/label-versions", json={}).status_code == 409  # nothing reviewed yet
    assert c.post("/proposals/accept", json={"min_confidence": 0.5}).json() == {"accepted": 1, "rejected": 1, "images": 1}
    lv = c.post("/label-versions", json={}).json()
    assert (lv["images"], lv["boxes"]) == (1, 1)
    anns = c.get(f"/images/{h}/annotations").json()
    ok = [a for a in anns if a["status"] == "accepted"]
    assert len(anns) == 2 and len(ok) == 1 and ok[0]["source"] == "auto:sam3" and (ok[0]["cx"], ok[0]["w"]) == (0.25, 0.5)
    # lowering the threshold later brings the rejected box back; raising it removes both
    assert c.post("/proposals/accept", json={"min_confidence": 0.1}).json()["accepted"] == 2
    assert c.post("/proposals/accept", json={"min_confidence": 0.95}).json()["accepted"] == 0


def test_overview_thumb_and_job(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    c = TestClient(api.create_app(db.make_engine(f"sqlite:///{tmp_path / 't.db'}"), detector=FakeDetector()))
    c.put("/classes", json=["car"])
    h = c.post("/images", files=[("files", ("a.png", png("red"), "image/png"))]).json()[0]["hash"]
    assert c.get("/overview").json()[0]["proposed"] == 0
    assert c.get(f"/images/{h}/thumb").headers["content-type"] == "image/jpeg"

    job = c.post("/jobs/autolabel", json={}).json()
    for _ in range(50):
        job = c.get(f"/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert (job["status"], job["done"], job["boxes"]) == ("done", 1, 2)
    o = c.get("/overview").json()[0]
    assert (o["proposed"], o["accepted"], o["reviewed"]) == (2, 0, False)

    c.post("/proposals/accept", json={"min_confidence": 0.5, "hashes": [h], "mark_reviewed": False})
    o = c.get("/overview").json()[0]
    assert (o["accepted"], o["rejected"], o["reviewed"]) == (1, 1, False)


def test_server_hosts_separate_datasets(tmp_path, monkeypatch):
    from labelhub import projects

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    c = TestClient(projects.create_server(tmp_path, detector=FakeDetector()))
    assert c.get("/api/projects").json() == []
    a = c.post("/api/projects", json={"name": "Street Cones"}).json()
    b = c.post("/api/projects", json={"name": "Street cones"}).json()
    assert (a["slug"], b["slug"]) == ("street-cones", "street-cones-2") and a["name"] == "Street Cones"
    assert c.post("/api/projects", json={"name": "  !! "}).status_code == 422

    c.put("/api/p/street-cones/classes", json=["cone"])
    c.put("/api/p/street-cones-2/classes", json=["bottle"])
    h = c.post("/api/p/street-cones/images", files=[("files", ("a.png", png("red"), "image/png"))]).json()[0]["hash"]
    assert c.get("/api/p/street-cones/classes").json() == ["cone"]
    assert c.get("/api/p/street-cones-2/classes").json() == ["bottle"]
    assert c.get("/api/p/street-cones-2/overview").json() == []  # images do not leak across datasets
    assert c.get(f"/api/p/street-cones/images/{h}/file").status_code == 200
    assert c.get(f"/api/p/street-cones-2/images/{h}/file").status_code == 404
    assert c.get("/api/p/nope/classes").status_code == 404
    assert (tmp_path / "projects" / "street-cones" / "images").exists()
    sums = {p["slug"]: p for p in c.get("/api/projects").json()}
    assert sums["street-cones"]["images"] == 1 and sums["street-cones-2"]["images"] == 0

    # one SAM3 shared by all datasets, and the job still sees the right dataset's folders
    job = c.post("/api/p/street-cones/jobs/autolabel", json={"min_confidence": 0.5}).json()
    import time
    for _ in range(50):
        job = c.get(f"/api/p/street-cones/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    ov = c.get("/api/p/street-cones/overview").json()[0]
    assert (ov["accepted"], ov["rejected"], ov["auto"]) == (1, 1, 2)


def test_legacy_database_in_root_is_a_dataset(tmp_path, monkeypatch):
    from labelhub import projects

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    engine = db.make_engine(f"sqlite:///{tmp_path / 'labelhub.db'}")
    TestClient(api.create_app(engine)).put("/classes", json=["car"])
    c = TestClient(projects.create_server(tmp_path))
    assert [p["slug"] for p in c.get("/api/projects").json()] == ["default"]
    assert c.get("/api/p/default/classes").json() == ["car"]


def test_view_settles_proposals_and_marks_reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    c = TestClient(api.create_app(db.make_engine(f"sqlite:///{tmp_path / 't.db'}"), detector=FakeDetector()))
    c.put("/classes", json=["car"])
    h = c.post("/images", files=[("files", ("a.png", png("red"), "image/png"))]).json()[0]["hash"]
    c.post("/autolabel", json={})  # legacy flow: proposals
    assert c.get("/overview").json()[0]["proposed"] == 2
    assert c.post(f"/images/{h}/view").json()["reviewed"] is True
    o = c.get("/overview").json()[0]
    assert (o["accepted"], o["rejected"], o["proposed"], o["reviewed"]) == (1, 1, 0, True)
    # default targets of a new auto-label are images with no auto boxes yet
    assert c.post("/autolabel", json={}).json() == {"images": 0, "boxes": 0}


def test_rerun_with_min_confidence_keeps_hand_drawn_boxes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    c = TestClient(api.create_app(db.make_engine(f"sqlite:///{tmp_path / 't.db'}"), detector=FakeDetector()))
    c.put("/classes", json=["car"])
    h = c.post("/images", files=[("files", ("a.png", png("red"), "image/png"))]).json()[0]["hash"]
    c.post("/autolabel", json={"min_confidence": 0.5})
    c.post(f"/images/{h}/annotations", json={"class_name": "car", "cx": .5, "cy": .5, "w": .1, "h": .1})
    c.post("/autolabel", json={"hashes": [h], "min_confidence": 0.5})  # redo SAM3 for this image
    anns = c.get(f"/images/{h}/annotations").json()
    assert sorted((a["source"], a["status"]) for a in anns) == [("auto:sam3", "accepted"), ("auto:sam3", "rejected"), ("human", "accepted")]


def test_image_without_detections_is_not_new_again(tmp_path, monkeypatch):
    class Nothing:
        def detect(self, image, classes):
            return []

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    c = TestClient(api.create_app(db.make_engine(f"sqlite:///{tmp_path / 't.db'}"), detector=Nothing()))
    c.put("/classes", json=["car"])
    c.post("/images", files=[("files", ("a.png", png("red"), "image/png"))])
    assert c.post("/autolabel", json={}).json() == {"images": 1, "boxes": 0}
    assert c.post("/autolabel", json={}).json() == {"images": 0, "boxes": 0}
