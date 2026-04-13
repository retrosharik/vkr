from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from adf_core_python.launcher import Launcher


def _load_system_settings() -> dict:
    config_path = Path('./config/runtime/system.json')
    try:
        return json.loads(config_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _clear_directory_contents(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    for entry in path.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except Exception:
            pass


def _cleanup_runtime_logs(settings: dict) -> None:
    logs_cfg = settings.get('logs', {}) if isinstance(settings, dict) else {}
    if not bool(logs_cfg.get('auto_clean_on_start', False)):
        return

    root = Path('.')
    for relative in (
        logs_cfg.get('raw_dir', 'runtime/raw_logs'),
        logs_cfg.get('my_dir', 'runtime/my_logs'),
        logs_cfg.get('debug_dir', 'runtime/log'),
    ):
        if not relative:
            continue
        _clear_directory_contents(root / str(relative))

    if bool(logs_cfg.get('clean_top_level_agent_logs', True)):
        for pattern in ('agent_*.log', 'agent.log'):
            for candidate in root.glob(pattern):
                try:
                    if candidate.is_dir():
                        shutil.rmtree(candidate)
                    else:
                        candidate.unlink()
                except Exception:
                    pass


def _should_clean_logs_on_start(settings: dict) -> bool:
    logs_cfg = settings.get('logs', {}) if isinstance(settings, dict) else {}
    if not bool(logs_cfg.get('auto_clean_on_start', False)):
        return False
    return os.environ.get('BASE_RESCUE_AGENT_CLEAN_LOGS_ON_START', '').strip().lower() in {'1', 'true', 'yes'}


if __name__ == '__main__':
    settings = _load_system_settings()
    if _should_clean_logs_on_start(settings):
        _cleanup_runtime_logs(settings)
    launcher = Launcher('./config/launcher.yaml')
    launcher.launch()
