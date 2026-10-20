import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select
from sqlalchemy import func

from . import api, config, db, training
from .db import Annotation, Image, ImageVersion, ModelVersion


@dataclass
class Dataset:
    slug: str
    dir: Path
    engine: object
    app: FastAPI


class NewDataset(BaseModel):
    name: str


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]


class Registry:
    """Datasets live in <root>/projects/<slug>/ (own database, images and models).
    A database directly in <root> (the layout before datasets existed) shows up as 'default'."""

    def __init__(self, root: Path, detector=None, launcher=None):
        self.root, self.launcher = root, launcher
        self.shared = api.new_shared(detector)
        self._cache: dict[str, Dataset] = {}
        self._lock = threading.Lock()

    def dirs(self) -> dict[str, Path]:
        found = {p.name: p for p in sorted((self.root / "projects").glob("*")) if (p / "labelhub.db").exists()}
        if (self.root / "labelhub.db").exists():
            found = {"default": self.root, **found}
        return found

    def get(self, slug: str) -> Dataset | None:
        dirs = self.dirs()
        if slug not in dirs:
            return None
        with self._lock:
            if slug not in self._cache:
                d = dirs[slug]
                engine = db.make_engine(f"sqlite:///{d / 'labelhub.db'}")
                app = api.create_app(engine, launcher=self.launcher, shared=self.shared, probe=self.probe)
                self._cache[slug] = Dataset(slug, d, engine, app)
            return self._cache[slug]

    def probe(self):
        """A training run in any dataset: (dataset slug, model name) or None."""
        for slug in self.dirs():
            ds = self.get(slug)
            with Session(ds.engine) as s:
                m = training.running(s)
            if m:
                return slug, m.name
        return None

    def name_of(self, slug: str) -> str:
        meta = self.dirs()[slug] / "meta.json"
        if meta.exists():
            return json.loads(meta.read_text()).get("name", slug)
        return "Default" if slug == "default" else slug

    def summary(self, slug: str) -> dict:
        ds = self.get(slug)
        with Session(ds.engine) as s:
            images = s.exec(select(Image).where(Image.removed == False)).all()  # noqa: E712
            boxes = s.exec(select(func.count()).select_from(Annotation).where(Annotation.status == "accepted")).one()
            models = s.exec(select(func.count()).select_from(ModelVersion)).one()
            versions = s.exec(select(func.count()).select_from(ImageVersion)).one()
        meta = ds.dir / "meta.json"
        created = json.loads(meta.read_text()).get("created_at") if meta.exists() else None
        return {
            "slug": slug,
            "name": self.name_of(slug),
            "images": len(images),
            "reviewed": sum(i.reviewed for i in images),
            "boxes": boxes,
            "models": models,
            "versions": versions,
            "cover": images[0].hash if images else None,
            "created_at": created,
        }

    def create(self, name: str) -> str:
        name = name.strip()
        slug = slugify(name)
        if not slug:
            raise ValueError("give the dataset a name (letters or digits)")
        taken = set(self.dirs())
        base, n = slug, 2
        while slug in taken:
            slug, n = f"{base}-{n}", n + 1
        d = self.root / "projects" / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({"name": name, "created_at": datetime.now(timezone.utc).isoformat()}))
        db.make_engine(f"sqlite:///{d / 'labelhub.db'}").dispose()
        return slug


class DatasetRouter:
    """ASGI app mounted at /api/p: /api/p/<slug>/... goes to that dataset's API."""

    def __init__(self, registry: Registry):
        self.registry = registry

    async def __call__(self, scope, receive, send):
        root_path = scope.get("root_path", "")
        path = scope["path"]
        rel = path[len(root_path):] if path.startswith(root_path) else path
        slug = rel.lstrip("/").split("/", 1)[0]
        ds = self.registry.get(slug)
        if not ds:
            await JSONResponse({"detail": f"dataset {slug!r} not found"}, status_code=404)(scope, receive, send)
            return
        token = config.use_data_dir(ds.dir)
        try:
            await ds.app({**scope, "root_path": f"{root_path}/{slug}"}, receive, send)
        finally:
            config.reset_data_dir(token)


def create_server(root: Path | None = None, detector=None, launcher=None) -> FastAPI:
    """All datasets under /api/p/<slug>, the dataset list under /api/projects, the built frontend at /."""
    registry = Registry(root or config.DATA_DIR, detector, launcher)
    app = FastAPI(title="LabelHub")

    @app.get("/api/projects")
    def list_projects():
        return [registry.summary(s) for s in registry.dirs()]

    @app.post("/api/projects")
    def create_project(body: NewDataset):
        try:
            return registry.summary(registry.create(body.name))
        except ValueError as e:
            raise HTTPException(422, str(e))

    app.mount("/api/p", DatasetRouter(registry))
    dist = config.ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True))
    return app
