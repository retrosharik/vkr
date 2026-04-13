from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PATH_EDGE_RISK_FEATURES = [
    'path_distance', 'node_count', 'road_nodes', 'building_nodes', 'expanded', 'stationary_ticks',
    'startup_recovery_locked', 'blocked_first_hops_active_count', 'skipped_start_edges_count',
    'failed_first_hop_count', 'first_hop_preblocked', 'first_hop_backtracks', 'same_goal_as_last_path',
    'phase_search', 'phase_transport', 'phase_move_to_victim', 'caller_action_move', 'caller_action_transport',
    'caller_action_clear', 'caller_command_executor', 'caller_search', 'caller_detector',
]

PATH_EDGE_RISK_V2_FEATURES = [
    'candidate_risk', 'candidate_base_cost', 'candidate_final_cost', 'candidate_distance', 'candidate_fail_count',
    'candidate_backtracks', 'candidate_blocked_history', 'candidate_count', 'candidate_rank_by_final_cost',
    'candidate_rank_by_risk', 'baseline_risk', 'baseline_base_cost', 'baseline_final_cost', 'baseline_risk_high',
    'is_baseline_first_hop', 'is_ml_best_first_hop', 'risk_improvement_vs_baseline', 'base_cost_delta_vs_baseline',
    'base_cost_ratio_vs_baseline', 'final_cost_delta_vs_baseline', 'final_cost_ratio_vs_baseline',
    'distance_delta_vs_baseline', 'distance_ratio_vs_baseline', 'stationary_ticks', 'startup_recovery_locked',
    'blocked_first_hops_active_count', 'skipped_start_edges_count', 'path_node_count', 'path_distance',
    'path_expanded', 'phase_search', 'phase_transport', 'phase_move_to_victim', 'caller_action_move',
    'caller_action_transport', 'caller_action_clear', 'caller_command_executor', 'caller_search', 'caller_detector',
]

PATH_EDGE_RISK_V3_FEATURES = list(PATH_EDGE_RISK_V2_FEATURES)


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


class PathEdgeRiskModel:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict):
            self.model = artifact['model']
            self.features = list(artifact.get('features', PATH_EDGE_RISK_FEATURES))
            self.metadata = dict(artifact.get('metadata', {}))
        else:
            self.model = artifact
            self.features = list(PATH_EDGE_RISK_FEATURES)
            self.metadata = {}
        self.version = str(self.metadata.get('version', 'path_edge_risk_v1'))
        self.model_name = type(self.model).__name__

    def describe(self) -> dict[str, Any]:
        return {
            'version': self.version,
            'model_name': self.model_name,
            'feature_count': len(self.features),
            'features': list(self.features),
        }

    def _context_features(self, merged: dict[str, Any]) -> None:
        phase = str(merged.get('phase') or '')
        merged['phase_search'] = 1.0 if phase == 'search' else 0.0
        merged['phase_transport'] = 1.0 if phase == 'transport' else 0.0
        merged['phase_move_to_victim'] = 1.0 if phase == 'move_to_victim' else 0.0
        caller_context = str(merged.get('caller_context') or '')
        merged['caller_action_move'] = 1.0 if caller_context == 'action_move' else 0.0
        merged['caller_action_transport'] = 1.0 if caller_context == 'action_transport' else 0.0
        merged['caller_action_clear'] = 1.0 if caller_context == 'action_clear' else 0.0
        merged['caller_command_executor'] = 1.0 if caller_context == 'command_executor' else 0.0
        merged['caller_search'] = 1.0 if caller_context == 'search' else 0.0
        merged['caller_detector'] = 1.0 if caller_context == 'detector' else 0.0

    def _build_v1_row(self, context: dict[str, Any], path_payload: dict[str, Any]) -> list[float]:
        merged = dict(context)
        merged.update(path_payload)
        self._context_features(merged)
        return [_safe_float(merged.get(feature)) for feature in self.features]

    def _build_v2_row(self, context: dict[str, Any], path_payload: dict[str, Any]) -> list[float]:
        merged = dict(context)
        merged.update(path_payload)
        self._context_features(merged)
        candidate_base_cost = _safe_float(merged.get('candidate_base_cost', merged.get('base_cost')))
        candidate_final_cost = _safe_float(merged.get('candidate_final_cost', merged.get('final_cost')))
        candidate_distance = _safe_float(merged.get('candidate_distance', merged.get('distance')))
        candidate_risk = _safe_float(merged.get('candidate_risk', merged.get('risk')))
        baseline_base_cost = _safe_float(merged.get('baseline_base_cost', 0.0))
        baseline_final_cost = _safe_float(merged.get('baseline_final_cost', 0.0))
        baseline_distance = _safe_float(merged.get('baseline_distance', merged.get('distance', 0.0)))
        baseline_risk = _safe_float(merged.get('baseline_risk', 0.0))
        merged.setdefault('candidate_base_cost', candidate_base_cost)
        merged.setdefault('candidate_final_cost', candidate_final_cost)
        merged.setdefault('candidate_distance', candidate_distance)
        merged.setdefault('candidate_risk', candidate_risk)
        merged.setdefault('baseline_base_cost', baseline_base_cost)
        merged.setdefault('baseline_final_cost', baseline_final_cost)
        merged.setdefault('baseline_risk', baseline_risk)
        merged.setdefault('risk_improvement_vs_baseline', baseline_risk - candidate_risk)
        merged.setdefault('base_cost_delta_vs_baseline', candidate_base_cost - baseline_base_cost)
        merged.setdefault('base_cost_ratio_vs_baseline', (candidate_base_cost / baseline_base_cost) if baseline_base_cost > 0 else 1.0)
        merged.setdefault('final_cost_delta_vs_baseline', candidate_final_cost - baseline_final_cost)
        merged.setdefault('final_cost_ratio_vs_baseline', (candidate_final_cost / baseline_final_cost) if baseline_final_cost > 0 else 1.0)
        merged.setdefault('distance_delta_vs_baseline', candidate_distance - baseline_distance)
        merged.setdefault('distance_ratio_vs_baseline', (candidate_distance / baseline_distance) if baseline_distance > 0 else 1.0)
        merged.setdefault('candidate_count', _safe_float(merged.get('candidate_count', 1.0)))
        merged.setdefault('candidate_rank_by_final_cost', _safe_float(merged.get('candidate_rank_by_final_cost', 1.0)))
        merged.setdefault('candidate_rank_by_risk', _safe_float(merged.get('candidate_rank_by_risk', 1.0)))
        merged.setdefault('candidate_fail_count', _safe_float(merged.get('candidate_fail_count', merged.get('fail_count', 0.0))))
        merged.setdefault('candidate_backtracks', _safe_float(merged.get('candidate_backtracks', merged.get('backtracks', 0.0))))
        merged.setdefault('candidate_blocked_history', _safe_float(merged.get('candidate_blocked_history', merged.get('blocked_history', 0.0))))
        merged.setdefault('baseline_risk_high', _safe_float(merged.get('baseline_risk_high', 0.0)))
        merged.setdefault('is_baseline_first_hop', _safe_float(merged.get('is_baseline_first_hop', 0.0)))
        merged.setdefault('is_ml_best_first_hop', _safe_float(merged.get('is_ml_best_first_hop', 0.0)))
        merged.setdefault('path_node_count', _safe_float(merged.get('path_node_count', merged.get('node_count', 0.0))))
        merged.setdefault('path_distance', _safe_float(merged.get('path_distance', merged.get('distance', 0.0))))
        merged.setdefault('path_expanded', _safe_float(merged.get('path_expanded', merged.get('expanded', 0.0))))
        return [_safe_float(merged.get(feature)) for feature in self.features]

    def _build_row(self, context: dict[str, Any], path_payload: dict[str, Any]) -> list[float]:
        return self._build_v2_row(context, path_payload) if self.version in {'path_edge_risk_v2', 'path_edge_risk_v3'} else self._build_v1_row(context, path_payload)

    def score_path(self, context: dict[str, Any], path_payload: dict[str, Any]) -> float | None:
        matrix = np.asarray([self._build_row(context, path_payload)], dtype=np.float64)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(matrix)
            positive_prob = float(probabilities[0, 1]) if probabilities.ndim == 2 and probabilities.shape[1] >= 2 else float(probabilities.reshape(-1)[0])
        elif hasattr(self.model, 'decision_function'):
            margin = float(np.asarray(self.model.decision_function(matrix), dtype=np.float64).reshape(-1)[0])
            positive_prob = float(1.0 / (1.0 + np.exp(-margin)))
        else:
            predictions = np.asarray(self.model.predict(matrix), dtype=np.float64).reshape(-1)
            positive_prob = float(predictions[0]) if predictions.size else None
        if positive_prob is None:
            return None
        return float(1.0 - positive_prob) if self.version in {'path_edge_risk_v2', 'path_edge_risk_v3'} else float(positive_prob)


def save_metadata_json(output_path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
