from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rcrscore.entities import EntityID

from .runtime_settings import project_root, setting


def entity_value(entity_id: Optional[EntityID]) -> str:
    if entity_id is None:
        return 'None'
    getter = getattr(entity_id, 'get_value', None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return str(entity_id)
    return str(entity_id)


class DecisionLogger:
    def __init__(self, agent_info, world_info, module_name: str, module_type: str) -> None:
        self.agent_info = agent_info
        self.world_info = world_info
        self.module_name = module_name
        self.module_type = module_type
        self.start_wall = time.monotonic()
        self.run_id = os.environ.get('RRS_BENCHMARK_RUN_ID') or f"manual__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.instance_label = f'i{id(self) % 100000}'
        root = project_root()
        raw_base = root / str(setting('logs.raw_dir', 'runtime/raw_logs')) / self.run_id
        my_base = root / str(setting('logs.my_dir', 'runtime/my_logs')) / self.run_id
        debug_base = root / str(setting('logs.debug_dir', 'runtime/log')) / self.run_id
        raw_base.mkdir(parents=True, exist_ok=True)
        my_base.mkdir(parents=True, exist_ok=True)
        debug_base.mkdir(parents=True, exist_ok=True)
        agent_id = entity_value(self._agent_id())
        self.raw_path = raw_base / f'{agent_id}__{module_type}.jsonl'
        self.text_path = my_base / f'{module_name}_{agent_id}.log'
        self.debug_json_path = debug_base / f'{module_name}_{agent_id}_{module_type}_{self.instance_label}.jsonl'
        self.debug_text_path = debug_base / f'{module_name}_{agent_id}_{module_type}_{self.instance_label}.log'
        logger_name = f'{module_name}.{module_type}.{agent_id}.{self.run_id}.{self.instance_label}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        formatter = logging.Formatter('%(message)s')
        handler = logging.FileHandler(self.text_path, encoding='utf-8')
        handler.setFormatter(formatter)
        self.logger.handlers.clear()
        self.logger.addHandler(handler)
        self.log_text('Модуль инициализирован', {'module': module_name, 'type': module_type, 'run_id': self.run_id, 'instance': self.instance_label})
        self.log_raw(None, 'module_loaded', {'module': module_name, 'type': module_type, 'run_id': self.run_id, 'instance': self.instance_label})
        self.debug_event('module_loaded', {'module': module_name, 'type': module_type, 'run_id': self.run_id, 'instance': self.instance_label})

    def _agent_id(self):
        getter = getattr(self.agent_info, 'get_entity_id', None)
        return getter() if callable(getter) else None

    def _tick(self) -> int:
        getter = getattr(self.agent_info, 'get_time', None)
        try:
            return int(getter()) if callable(getter) else 0
        except Exception:
            return 0

    def _position_id(self):
        getter = getattr(self.agent_info, 'get_position_entity_id', None)
        try:
            return getter() if callable(getter) else None
        except Exception:
            return None

    def _prefix(self) -> str:
        return f"[тик={self._tick()} агент={entity_value(self._agent_id())} зона={entity_value(self._position_id())} inst={self.instance_label}]"

    def _build_tag(self) -> str:
        tag = setting('system.build_tag', None)
        if tag is None or str(tag).strip() == '':
            tag = setting('logs.build_tag', 'production')
        return str(tag)

    def _normalize_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {}
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, float):
                normalized[key] = round(value, 4)
            else:
                normalized[key] = value
        return normalized

    def log_text(self, message: str, payload: dict[str, Any] | None = None) -> None:
        payload = self._normalize_payload(payload)
        line = self._prefix() + ' ' + message
        if payload:
            body = ' '.join(f'{key}={payload[key]}' for key in sorted(payload))
            line = line + ' | ' + body
        self.logger.info(line)

    def log_raw(self, tick: int | None, event_type: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            'schema_version': int(setting('logs.schema_version', 2)),
            'run_id': self.run_id,
            'module': self.module_name,
            'module_type': self.module_type,
            'logger_instance': self.instance_label,
            'build_tag': self._build_tag(),
            'agent_id': entity_value(self._agent_id()),
            'tick': self._tick() if tick is None else tick,
            'event_type': event_type,
            'payload': payload or {},
        }
        with self.raw_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()

    def debug_text(self, message: str, payload: dict[str, Any] | None = None) -> None:
        payload = self._normalize_payload(payload)
        line = self._prefix() + ' ' + message
        if payload:
            body = ' '.join(f'{key}={payload[key]}' for key in sorted(payload))
            line = line + ' | ' + body
        with self.debug_text_path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')
            f.flush()

    def debug_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            'schema_version': int(setting('logs.schema_version', 2)),
            'run_id': self.run_id,
            'module': self.module_name,
            'module_type': self.module_type,
            'logger_instance': self.instance_label,
            'build_tag': self._build_tag(),
            'agent_id': entity_value(self._agent_id()),
            'tick': self._tick(),
            'wall_elapsed': round(self.wall_elapsed(), 6),
            'event_type': event_type,
            'payload': payload or {},
        }
        with self.debug_json_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()

    def decision_snapshot(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[dict[str, Any]],
        selected_id: str | None,
        selected_reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        limit = int(setting('logs.max_logged_candidates', 64))
        enriched: list[dict[str, Any]] = []
        selected_rank = None
        selected_heuristic_rank = None
        selected_ml_rank = None
        selected_final_rank = None
        for index, candidate in enumerate(candidates, start=1):
            item = dict(candidate)
            item['rank'] = index
            item['final_rank'] = item.get('final_rank', index)
            item['is_selected'] = selected_id is not None and str(item.get('candidate_id')) == str(selected_id)
            if item['is_selected']:
                if selected_rank is None:
                    selected_rank = index
                if selected_heuristic_rank is None and item.get('heuristic_rank') is not None:
                    selected_heuristic_rank = item.get('heuristic_rank')
                if selected_ml_rank is None and item.get('ml_rank') is not None:
                    selected_ml_rank = item.get('ml_rank')
                if selected_final_rank is None and item.get('final_rank') is not None:
                    selected_final_rank = item.get('final_rank')
            enriched.append(item)
        trimmed = enriched[:limit]
        meta = metadata or {}
        payload = {
            'decision_type': decision_type,
            'state': state,
            'candidates': trimmed,
            'selected_id': selected_id,
            'selected_reason': selected_reason,
            'selected_rank': selected_rank,
            'selected_rank_by_heuristic': selected_heuristic_rank,
            'selected_rank_by_ml': selected_ml_rank,
            'selected_rank_by_final': selected_final_rank,
            'selection_mode': meta.get('selection_mode'),
            'selected_by': meta.get('selected_by'),
            'exploration_used': bool(meta.get('exploration_used', False)),
            'top_k_candidates': meta.get('top_k_candidates', []),
            'candidate_count': len(candidates),
            'logged_candidate_count': len(trimmed),
            'metadata': meta,
        }
        self.log_raw(None, 'decision_snapshot', payload)
        self.debug_event('decision_snapshot', payload)

    def path_snapshot(self, request: dict[str, Any], result: dict[str, Any]) -> None:
        payload = {'request': request, 'result': result}
        self.log_raw(None, 'path_snapshot', payload)
        self.debug_event('path_snapshot', payload)

    def state_snapshot(self, name: str, payload: dict[str, Any]) -> None:
        self.log_raw(None, name, payload)
        self.debug_event(name, payload)

    def wall_elapsed(self) -> float:
        return time.monotonic() - self.start_wall

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(str(msg), *args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.logger.info(str(msg), *args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(str(msg), *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.logger.error(str(msg), *args, **kwargs)

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(str(msg), *args, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(str(msg), *args, **kwargs)
