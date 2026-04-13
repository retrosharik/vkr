from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'main.py').exists() and (parent / 'config').exists():
            return parent
    return Path.cwd()


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def runtime_settings() -> dict[str, Any]:
    config_root = project_root() / 'config'
    runtime_dir = config_root / 'runtime'
    merged: dict[str, Any] = {}
    if runtime_dir.exists() and runtime_dir.is_dir():
        preferred = ['system.json', 'search.json', 'detector.json', 'path.json']
        files = [runtime_dir / name for name in preferred if (runtime_dir / name).exists()]
        known = {path.name for path in files}
        files.extend(sorted(path for path in runtime_dir.glob('*.json') if path.name not in known))
        for path in files:
            merged = _deep_merge(merged, _load_json(path))
        return merged
    runtime_file = config_root / 'runtime.json'
    if runtime_file.exists():
        return _load_json(runtime_file)
    return merged


def setting(path: str, default: Any = None) -> Any:
    data: Any = runtime_settings()
    for part in path.split('.'):
        if not isinstance(data, dict) or part not in data:
            return default
        data = data[part]
    return data
