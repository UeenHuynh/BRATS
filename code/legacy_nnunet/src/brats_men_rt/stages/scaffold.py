from __future__ import annotations

import argparse
from pathlib import Path

from brats_men_rt.paths import project_root
from brats_men_rt.utils import touch, write_csv, write_json, write_text


def run_scaffold(stage_name: str, outputs: list[tuple[str, str]]) -> dict[str, object]:
    root = project_root()
    created = []
    for relative_path, kind in outputs:
        path = root / relative_path
        if kind == "json":
            if not path.exists():
                write_json(path, {"stage": stage_name, "status": "scaffold", "notes": "Implementation pending."})
        elif kind == "csv":
            if not path.exists():
                write_csv(path, [], ("placeholder",))
        elif kind == "md":
            if not path.exists():
                write_text(path, f"# {stage_name}\n\nScaffold created. Fill in implementation and results after the prerequisite stages complete.")
        else:
            touch(path)
        created.append(relative_path)
    return {"stage": stage_name, "created": created}


def main(stage_name: str, outputs: list[tuple[str, str]]) -> None:
    parser = argparse.ArgumentParser(description=f"Create scaffold outputs for {stage_name}.")
    parser.parse_args()
    result = run_scaffold(stage_name, outputs)
    print(f"Scaffolded {len(result['created'])} outputs for {result['stage']}")


if __name__ == "__main__":
    raise SystemExit("Use this module through a stage wrapper.")
