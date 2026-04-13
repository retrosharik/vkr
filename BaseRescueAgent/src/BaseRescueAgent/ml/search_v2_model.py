from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

SEARCH_V2_CANDIDATE_FEATURES = [
    'distance',
    'centrality',
    'civilian_hint_count',
    'active_civilians_inside',
    'buried_inside',
    'in_cluster',
    'visited',
    'recent_target',
    'current_target',
    'blocked',
    'path_nodes',
    'reachable',
    'candidate_count',
    'cluster_remaining_ratio',
    'known_civilians',
    'known_refuges',
    'cluster_candidate_count',
    'global_candidate_count',
    'cluster_unvisited_count',
    'outside_unvisited_count',
    'cluster_revisit_count',
    'outside_revisit_count',
    'forced_global',
    'scope_cluster_unvisited',
    'scope_outside_unvisited',
    'scope_cluster_revisit',
    'scope_outside_revisit',
    'phase_search',
    'phase_transport',
    'phase_move_to_victim',
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric = float(value)
        if not np.isfinite(numeric):
            return default
        return numeric
    except Exception:
        return default


class SearchV2Model:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict):
            self.model = artifact['model']
            self.features = list(artifact.get('features', SEARCH_V2_CANDIDATE_FEATURES))
            self.metadata = dict(artifact.get('metadata', {}))
        else:
            self.model = artifact
            self.features = list(SEARCH_V2_CANDIDATE_FEATURES)
            self.metadata = {}
        self.version = str(self.metadata.get('version', 'search_v2'))
        self.model_name = type(self.model).__name__

    def describe(self) -> dict[str, Any]:
        return {
            'version': self.version,
            'model_name': self.model_name,
            'feature_count': len(self.features),
            'features': list(self.features),
        }

    def _scope_features(self, merged: dict[str, Any]) -> None:
        scope = str(merged.get('scope') or merged.get('search_scope') or '')
        merged['scope_cluster_unvisited'] = 1.0 if scope == 'cluster_unvisited' else 0.0
        merged['scope_outside_unvisited'] = 1.0 if scope == 'outside_unvisited' else 0.0
        merged['scope_cluster_revisit'] = 1.0 if scope == 'cluster_revisit' else 0.0
        merged['scope_outside_revisit'] = 1.0 if scope == 'outside_revisit' else 0.0
        phase = str(merged.get('phase') or '')
        merged['phase_search'] = 1.0 if phase == 'search' else 0.0
        merged['phase_transport'] = 1.0 if phase == 'transport' else 0.0
        merged['phase_move_to_victim'] = 1.0 if phase == 'move_to_victim' else 0.0

    def _build_row(self, context: dict[str, Any], candidate: dict[str, Any]) -> list[float]:
        merged = dict(context)
        merged.update(candidate)
        self._scope_features(merged)
        return [_safe_float(merged.get(feature)) for feature in self.features]

    def score_candidates(self, context: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, float]:
        if not candidates:
            return {}
        matrix = np.asarray([self._build_row(context, candidate) for candidate in candidates], dtype=np.float64)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(matrix)
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                scores = probabilities[:, 1]
            else:
                scores = probabilities.reshape(-1)
        elif hasattr(self.model, 'decision_function'):
            margins = np.asarray(self.model.decision_function(matrix), dtype=np.float64).reshape(-1)
            scores = 1.0 / (1.0 + np.exp(-margins))
        else:
            scores = np.asarray(self.model.predict(matrix), dtype=np.float64).reshape(-1)
        return {
            str(candidate.get('candidate_id')): float(score)
            for candidate, score in zip(candidates, scores)
            if candidate.get('candidate_id') is not None
        }


def save_metadata_json(output_path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
