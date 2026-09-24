from __future__ import annotations

import argparse
import importlib


REQUIRED_MODULES = [
    "acvl_utils",
    "annotated_types",
    "batchgenerators",
    "batchgeneratorsv2",
    "blosc2",
    "dynamic_network_architectures",
    "einops",
    "graphviz",
    "imagecodecs",
    "matplotlib",
    "msgpack",
    "ndindex",
    "nibabel",
    "nnunetv2",
    "numexpr",
    "numpy",
    "pandas",
    "pydantic",
    "pydantic_core",
    "requests",
    "scipy",
    "seaborn",
    "SimpleITK",
    "skimage",
    "sklearn",
    "tifffile",
    "threadpoolctl",
    "torch",
    "tqdm",
    "typing_extensions",
    "typing_inspection",
    "yaml",
    "yacs",
]


def _module_status(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - direct environment probe
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "unknown")
    return True, str(version)


def _install_hints(missing_names: list[str]) -> list[str]:
    missing = set(missing_names)
    hints: list[str] = []

    if "torch" in missing:
        hints.append(
            "Torch is missing. Install PyTorch separately with a command matched to your CUDA/CPU setup from pytorch.org."
        )

    if missing & {
        "acvl_utils",
        "annotated_types",
        "batchgenerators",
        "batchgeneratorsv2",
        "blosc2",
        "dynamic_network_architectures",
        "graphviz",
        "msgpack",
        "ndindex",
        "nnunetv2",
        "numexpr",
        "pydantic",
        "pydantic_core",
        "threadpoolctl",
        "typing_extensions",
        "typing_inspection",
    }:
        hints.append("Re-run:\n  pip install --no-deps -r requirements.txt")

    if "graphviz" in missing:
        hints.append(
            "If `import graphviz` succeeds later but rendering still fails, install the Graphviz binary with:\n  conda install -c conda-forge graphviz"
        )

    return hints


def run() -> dict[str, object]:
    missing = []
    found = []
    for name in REQUIRED_MODULES:
        ok, detail = _module_status(name)
        if ok:
            found.append((name, detail))
        else:
            missing.append((name, detail))

    print("nnU-Net environment check")
    print("")
    print("Found modules:")
    for name, detail in found:
        print(f"  OK      {name:<32} {detail}")

    print("")
    print("Missing/broken modules:")
    if not missing:
        print("  none")
    else:
        for name, detail in missing:
            print(f"  MISSING {name:<32} {detail}")

    hints = _install_hints([name for name, _ in missing])
    for hint in hints:
        print("")
        print(hint)

    return {"missing": [name for name, _ in missing], "found": [name for name, _ in found]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the nnU-Net environment is complete.")
    parser.parse_args()
    result = run()
    print("")
    print(f"Summary: found={len(result['found'])} missing={len(result['missing'])}")


if __name__ == "__main__":
    main()
