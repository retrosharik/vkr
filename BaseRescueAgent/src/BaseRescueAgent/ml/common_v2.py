from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_jsonl_from_directory(root: Path, suffixes: tuple[str, ...] = ('.jsonl',)) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in sorted(root.rglob('*')):
        if not path.is_file() or not path.name.endswith(suffixes):
            continue
        relative = str(path.relative_to(root))
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield relative, json.loads(line)


def iter_jsonl_from_zip(path: Path, suffixes: tuple[str, ...] = ('.jsonl',)) -> Iterator[tuple[str, dict[str, Any]]]:
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(suffixes):
                continue
            with zf.open(name) as f:
                for raw_line in f:
                    line = raw_line.decode('utf-8').strip()
                    if line:
                        yield name, json.loads(line)


def iter_records(input_path: Path, suffixes: tuple[str, ...] = ('.jsonl',)) -> Iterator[tuple[str, dict[str, Any]]]:
    if input_path.is_dir():
        yield from iter_jsonl_from_directory(input_path, suffixes=suffixes)
        return
    if input_path.suffix.lower() == '.zip':
        yield from iter_jsonl_from_zip(input_path, suffixes=suffixes)
        return
    raise ValueError(f'Unsupported input path: {input_path}')


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(float(value))
    except Exception:
        return default


def text_value(value: Any, default: str = '') -> str:
    if value is None:
        return default
    return str(value)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_allowed_modes(raw: str, default_modes: tuple[str, ...]) -> set[str]:
    parts = [item.strip() for item in str(raw).split(',') if item.strip()]
    return set(parts) if parts else set(default_modes)


def decision_key(run_id: str, agent_id: str, tick: int, selected_id: str, decision_type: str) -> tuple[str, str, int, str, str]:
    return (str(run_id), str(agent_id), int(tick), str(selected_id), str(decision_type))
