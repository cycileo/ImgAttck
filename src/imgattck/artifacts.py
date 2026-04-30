from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from imgattck.config import dump_config


def create_run_dir(config: Any, kind: str) -> Path:
    root = Path(config.output.root)
    name = config.output.name or f"{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def snapshot_config(run_dir: Path, config: Any) -> None:
    dump_config(config, run_dir / "config.yaml")
