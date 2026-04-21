from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from .runtime_settings import project_root, setting

try:
    from ...ml.detector_v3_model import DetectorV3Model
except Exception:
    DetectorV3Model = None

try:
    from ...ml.search_v2_model import SearchV2Model
except Exception:
    SearchV2Model = None

try:
    from ...ml.path_edge_risk_model import PathEdgeRiskModel
except Exception:
    PathEdgeRiskModel = None


_MODEL_CACHE: dict[str, Any] = {}
_MODEL_ERRORS: dict[str, str] = {}
_MODEL_LOCK = Lock()


class MlBridge:
    def __init__(self, module_type: str) -> None:
        self.module_type = module_type
        configured_mode = str(setting(f'{module_type}.mode', 'heuristic') or 'heuristic').strip().lower()
        self.use_ml_flag = bool(setting(f'{module_type}.use_ml', False))
        if configured_mode not in {'heuristic', 'hybrid', 'pure_ml_test', 'shadow'}:
            configured_mode = 'heuristic'
        if not self.use_ml_flag and configured_mode != 'pure_ml_test':
            configured_mode = 'heuristic'
        elif configured_mode == 'heuristic' and self.use_ml_flag:
            configured_mode = 'hybrid'
        self.requested_mode = configured_mode
        self.model_path = setting(f'{module_type}.model_path', None)

    def _default_model_candidates(self) -> list[Path]:
        configured = setting(f'{self.module_type}.ml.auto_model_paths', [])
        paths: list[Path] = []
        if isinstance(configured, list):
            for value in configured:
                if not value:
                    continue
                path = Path(str(value))
                if not path.is_absolute():
                    path = project_root() / path
                resolved = path.resolve()
                if resolved not in paths:
                    paths.append(resolved)
        fallback_paths: list[Path] = []
        if self.module_type == 'detector':
            fallback_paths = [
                project_root() / 'models' / 'stage4_detector_v3_from_65runs' / 'detector_v3.joblib',
            ]
        elif self.module_type == 'search':
            fallback_paths = [
                project_root() / 'models' / 'stage3_v2' / 'search_v2' / 'search_v2.joblib',
            ]
        elif self.module_type == 'path':
            fallback_paths = [
                project_root() / 'models' / 'path_edge_risk_v1' / 'path_edge_risk_v1.joblib',
            ]
        for path in fallback_paths:
            resolved = path.resolve()
            if resolved not in paths:
                paths.append(resolved)
        return paths

    def _resolved_model_path(self) -> Path | None:
        checked: list[Path] = []
        if self.model_path:
            path = Path(str(self.model_path))
            if not path.is_absolute():
                path = project_root() / path
            resolved = path.resolve()
            checked.append(resolved)
            if resolved.exists():
                return resolved
        for candidate in self._default_model_candidates():
            if candidate in checked:
                continue
            if candidate.exists():
                return candidate
        return checked[0] if checked else None

    def _load_model(self) -> Any | None:
        path = self._resolved_model_path()
        if path is None:
            return None
        cache_key = f'{self.module_type}:{path}'
        with _MODEL_LOCK:
            if cache_key in _MODEL_CACHE:
                return _MODEL_CACHE[cache_key]
            if cache_key in _MODEL_ERRORS:
                return None
            try:
                if not path.exists():
                    raise FileNotFoundError(str(path))
                if self.module_type == 'detector':
                    if DetectorV3Model is None:
                        raise RuntimeError('DetectorV3Model import failed')
                    model = DetectorV3Model(path)
                elif self.module_type == 'search':
                    if SearchV2Model is None:
                        raise RuntimeError('SearchV2Model import failed')
                    model = SearchV2Model(path)
                elif self.module_type == 'path':
                    if PathEdgeRiskModel is None:
                        raise RuntimeError('PathEdgeRiskModel import failed')
                    model = PathEdgeRiskModel(path)
                else:
                    raise RuntimeError(f'Unsupported ML module type: {self.module_type}')
                _MODEL_CACHE[cache_key] = model
                return model
            except Exception as exc:
                _MODEL_ERRORS[cache_key] = str(exc)
                return None

    def is_requested(self) -> bool:
        return self.requested_mode in {'hybrid', 'pure_ml_test', 'shadow'}

    def is_active(self) -> bool:
        return self.is_requested() and self._load_model() is not None

    def mode_name(self) -> str:
        if not self.is_requested():
            return 'heuristic'
        if not self.is_active():
            return 'heuristic_fallback'
        return self.requested_mode

    def is_shadow_mode(self) -> bool:
        return self.mode_name() == 'shadow'

    def describe(self) -> dict[str, Any]:
        resolved = self._resolved_model_path()
        configured_path = str(self.model_path) if self.model_path else None
        info: dict[str, Any] = {
            'module_type': self.module_type,
            'requested_mode': self.requested_mode,
            'active_mode': self.mode_name(),
            'configured_model_path': configured_path,
            'resolved_model_path': str(resolved) if resolved is not None else None,
            'requested_ml': self.is_requested(),
            'active': self.is_active(),
        }
        if resolved is not None:
            try:
                stat = resolved.stat()
                info['model_exists'] = resolved.exists()
                info['model_size_bytes'] = int(stat.st_size)
                info['model_mtime'] = int(stat.st_mtime)
            except Exception:
                info['model_exists'] = resolved.exists()
        if not self.is_requested():
            return info
        model = self._load_model()
        if model is not None and hasattr(model, 'describe'):
            try:
                info['model'] = model.describe()
            except Exception:
                pass
        cache_key = f'{self.module_type}:{resolved}' if resolved is not None else None
        if cache_key is not None and cache_key in _MODEL_ERRORS:
            info['load_error'] = _MODEL_ERRORS[cache_key]
        return info

    def score_candidates(self, context: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, float]:
        if not self.is_requested() or not candidates:
            return {}
        model = self._load_model()
        if model is None:
            return {}
        if hasattr(model, 'score_candidates'):
            try:
                scores = model.score_candidates(context, candidates)
                return {str(key): float(value) for key, value in scores.items()}
            except Exception:
                return {}
        return {}

    def score_path(self, context: dict[str, Any], path_payload: dict[str, Any]) -> float | None:
        if not self.is_requested():
            return None
        model = self._load_model()
        if model is None:
            return None
        if hasattr(model, 'score_path'):
            try:
                value = model.score_path(context, path_payload)
                return None if value is None else float(value)
            except Exception:
                return None
        return None
