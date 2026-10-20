from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine

from . import config


class Image(SQLModel, table=True):
    hash: str = Field(primary_key=True)
    filename: str
    ext: str
    width: int
    height: int
    removed: bool = False
    reviewed: bool = False
    auto_done: bool = False
    gold: bool = False


class WsClass(SQLModel, table=True):
    name: str = Field(primary_key=True)
    position: int


class Annotation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    image_hash: str = Field(index=True, foreign_key="image.hash")
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    source: str = "human"
    status: str = "accepted"
    confidence: float | None = None


class _Version(SQLModel):
    name: str = Field(primary_key=True)
    parent: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note: str = ""
    hash: str


class ImageVersion(_Version, table=True):
    media_json: str
    split_json: str


class LabelVersion(_Version, table=True):
    classes_json: str
    labels_json: str
    stats_json: str = "{}"


class ModelVersion(SQLModel, table=True):
    name: str = Field(primary_key=True)
    parent: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note: str = ""
    image_version: str
    label_version: str
    base: str
    config_json: str = "{}"
    status: str = "running"  # running | done | failed | cancelled
    epoch: int = 0
    epochs: int = 0
    metrics_json: str = "{}"
    error: str | None = None
    pid: int | None = None
    imported: bool = False


class InferenceLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model: str = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_ms: float
    infer_ms: float
    device: str
    cold: bool = False
    boxes: int = 0
    width: int = 0
    height: int = 0
    source: str = "ui"


def make_engine(url: str | None = None):
    url = url or config.DB_URL
    Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    _migrate(engine)
    return engine


def _migrate(engine) -> None:
    """Add columns introduced after a database was first created."""
    with engine.begin() as conn:
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(labelversion)")}
        if cols and "stats_json" not in cols:
            conn.exec_driver_sql("ALTER TABLE labelversion ADD COLUMN stats_json VARCHAR DEFAULT '{}'")
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(image)")}
        if cols and "gold" not in cols:
            conn.exec_driver_sql("ALTER TABLE image ADD COLUMN gold BOOLEAN DEFAULT 0")
        if cols and "auto_done" not in cols:
            conn.exec_driver_sql("ALTER TABLE image ADD COLUMN auto_done BOOLEAN DEFAULT 0")
            conn.exec_driver_sql("UPDATE image SET auto_done = 1 WHERE hash IN (SELECT image_hash FROM annotation WHERE source LIKE 'auto:%')")


def session(engine) -> Session:
    return Session(engine)
