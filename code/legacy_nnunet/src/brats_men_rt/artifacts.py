from __future__ import annotations

from pathlib import Path

from brats_men_rt.constants import ARTIFACT_MANIFEST_COLUMNS
from brats_men_rt.utils import utc_timestamp, write_csv


def init_artifact_manifest(path: Path) -> None:
    if path.exists():
        return
    write_csv(path, [], ARTIFACT_MANIFEST_COLUMNS)


def artifact_row(
    artifact_id: str,
    artifact_type: str,
    path: str,
    created_by_stage: str,
    input_dependencies: str = "",
    config_used: str = "",
    checksum: str = "",
    description: str = "",
    status: str = "planned",
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "path": path,
        "created_by_stage": created_by_stage,
        "input_dependencies": input_dependencies,
        "config_used": config_used,
        "timestamp": utc_timestamp(),
        "checksum": checksum,
        "description": description,
        "status": status,
    }
