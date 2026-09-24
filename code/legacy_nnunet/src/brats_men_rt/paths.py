from __future__ import annotations

from pathlib import Path

from brats_men_rt.constants import PROJECT_DIRECTORIES


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return project_root() / "data" / "extracted"


def ensure_project_directories(root: Path | None = None) -> list[Path]:
    base = root or project_root()
    created = []
    for relative in PROJECT_DIRECTORIES:
        path = base / relative
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def relative_to_root(path: Path, root: Path | None = None) -> str:
    base = root or project_root()
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())
