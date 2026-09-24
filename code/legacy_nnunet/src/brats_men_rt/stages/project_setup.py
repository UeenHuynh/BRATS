from __future__ import annotations

import argparse
from pathlib import Path

from brats_men_rt.artifacts import init_artifact_manifest
from brats_men_rt.paths import ensure_project_directories, project_root, relative_to_root
from brats_men_rt.utils import touch, write_json, write_text


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        write_text(path, content)


def _json_if_missing(path: Path, payload: dict[str, object]) -> None:
    if not path.exists():
        write_json(path, payload)


def run(root: Path | None = None) -> dict[str, object]:
    base = root or project_root()
    ensure_project_directories(base)

    _write_if_missing(
        base / "data_raw" / "README.md",
        "# data_raw\n\nRaw data is currently stored in `data/extracted/`. This folder exists to preserve the planned pipeline layout without copying sensitive or large data.",
    )
    _write_if_missing(
        base / "configs" / "environment.yml",
        """name: brats-men-rt-pipeline
channels:
  - conda-forge
dependencies:
  - python=3.11
  - libgcc-ng>=13
  - libstdcxx-ng>=13
  - numpy
  - pandas
  - scipy
  - matplotlib
  - pyyaml
  - nibabel
  - scikit-learn
  - simpleitk
  - tifffile
  - imagecodecs
  - seaborn
  - requests
  - graphviz
  - scikit-image
  - tqdm
  - pip
""",
    )
    _write_if_missing(
        base / "requirements.txt",
        """# Pip-only packages for the BRATS nnU-Net workflow.
# Install into the conda env with:
#   pip install --no-deps -r requirements.txt
# Runtime deps for packages such as `blosc2` are listed explicitly here because
# this repo installs with `--no-deps` to avoid pip mutating the conda solve.
acvl-utils>=0.2.6,<0.3
batchgenerators>=0.25.1
batchgeneratorsv2>=0.3.2
blosc2
dynamic-network-architectures>=0.4.4,<0.5
einops
graphviz
msgpack
ndindex
nnunetv2>=2.5
numexpr>=2.14.1
annotated-types>=0.6.0
pydantic
pydantic-core==2.47.0
threadpoolctl
typing-extensions>=4.15.0
typing-inspection>=0.4.2
yacs
""",
    )
    _json_if_missing(
        base / "configs" / "project_config.json",
        {
            "project_name": "BRATS-MEN-RT",
            "raw_data_root": "data/extracted",
            "processed_data_root": "data_processed",
            "labeled_training_split": "train",
            "unlabeled_validation_split": "val_unlabeled",
            "random_seed": 42,
        },
    )
    _json_if_missing(
        base / "configs" / "reproducibility_config.json",
        {
            "random_seed": 42,
            "timezone": "UTC",
            "command_log": "logs/commands.log",
            "artifact_manifest": "manifests/artifact_manifest.csv",
        },
    )
    _json_if_missing(
        base / "configs" / "preprocessing_config.json",
        {
            "target_orientation": "RAS",
            "resampling": {"enabled": False, "target_spacing_mm": [1.0, 1.0, 1.0]},
            "intensity_clipping_percentiles": [0.5, 99.5],
            "normalization_region": "nonzero",
            "crop_margin_voxels": [8, 8, 8],
        },
    )
    _json_if_missing(
        base / "configs" / "split_config.json",
        {
            "strategy": "five_fold_cv",
            "n_folds": 5,
            "stratify_by": "volume_group",
            "random_seed": 42,
        },
    )
    _json_if_missing(
        base / "configs" / "soft_label_config.json",
        {"boundary_width_mm": 2.0, "distance_mode": "signed_distance_transform"},
    )
    _json_if_missing(
        base / "hitl" / "hitl_simulation_config.json",
        {
            "strategies": ["auto_only", "random", "uncertainty_guided", "oracle"],
            "correction_steps": [1, 3, 5],
            "prompt_types": ["positive_click", "negative_click", "scribble", "bounding_box"],
        },
    )
    _write_if_missing(
        base / "configs" / "nnunet_baseline.yaml",
        """model: nnunet_v2_3d
patch_size: [128, 128, 128]
batch_size: 1
mixed_precision: true
loss: dice_ce
uncertainty_method: tta
""",
    )
    touch(base / "logs" / "commands.log")
    touch(base / "logs" / "preprocessing_log.jsonl")
    init_artifact_manifest(base / "manifests" / "artifact_manifest.csv")
    return {"root": relative_to_root(base, base), "status": "initialized"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize project folders and config templates.")
    parser.parse_args()
    result = run()
    print(f"Project scaffold initialized at {result['root'] or '.'}")


if __name__ == "__main__":
    main()
