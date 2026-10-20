import os
from contextvars import ContextVar
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("LABELHUB_DATA", ROOT / "data"))
IMAGES_DIR = DATA_DIR / "images"
DB_URL = f"sqlite:///{DATA_DIR / 'labelhub.db'}"

_current: ContextVar[Path | None] = ContextVar("labelhub_dataset_dir", default=None)


def use_data_dir(path: Path):
    """Point this request (and threads copied from its context) at one dataset's directory."""
    return _current.set(path)


def reset_data_dir(token) -> None:
    _current.reset(token)


def data_dir() -> Path:
    return _current.get() or DATA_DIR


def images_dir() -> Path:
    cur = _current.get()
    return cur / "images" if cur else IMAGES_DIR
