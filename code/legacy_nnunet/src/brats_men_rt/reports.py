from __future__ import annotations

from pathlib import Path

from brats_men_rt.utils import write_text


def write_markdown_report(path: Path, title: str, lines: list[str]) -> None:
    body = "\n".join(lines)
    write_text(path, f"# {title}\n\n{body}")
