
from __future__ import annotations

import random
import time
from typing import Any, Optional, cast

from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from adf_core_python.core.component.module.complex.human_detector import HumanDetector
from rcrscore.entities import Building, Civilian, EntityID, Human, Refuge

from ..util.decision_logger import DecisionLogger, entity_value
from ..util.ml_bridge import MlBridge
from ..util.rescue_support import (
    area_distance,
    cluster_entities_for_agent,
    count_ambulances_at,
    estimate_life_margin,
    is_rescuable_civilian,
    nearest_refuge,
    refuge_entities,
    searchable_buildings,
    urgency_score,
    visible_civilians,
)
from ..util.runtime_settings import setting
from ..util.shared_runtime_state import get_runtime_state


class PriorityHumanDetector(HumanDetector):
    def __init__(
        self,
        agent_info: AgentInfo,
        world_info: WorldInfo,
        scenario_info: ScenarioInfo,
        module_manager: ModuleManager,
        develop_data: DevelopData,
    ) -> None:
        super().__init__(agent_info, world_info, scenario_info, module_manager, develop_data)
        self._clustering: Clustering = cast(
            Clustering,
            module_manager.get_module(
                'PriorityHumanDetector.Clustering',
                'adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering',
            ),
        )
        self._path_planning: PathPlanning = cast(
            PathPlanning,
            module_manager.get_module(
                'PriorityHumanDetector.PathPlanning',
                'src.BaseRescueAgent.module.algorithm.logged_a_star_path_planning.LoggedAStarPathPlanning',
            ),
        )
        self.register_sub_module(self._clustering)
        self.register_sub_module(self._path_planning)
        self._result: Optional[EntityID] = None
        self._decision_logger = DecisionLogger(agent_info, world_info, self.__class__.__name__, 'detector')
        self._ml = MlBridge('detector')

    def update_info(self, message_manager: MessageManager) -> HumanDetector:
        super().update_info(message_manager)
        return self

    def _current_target_age(self, runtime_state, tick: int) -> int:
        if runtime_state.detector_target_since < 0:
            return 0
        return max(tick - runtime_state.detector_target_since, 0)

    def _candidate_features(self, civilian: Civilian, cluster_ids: set[EntityID], runtime_state, tick: int) -> dict[str, float | int | str | bool | None]:
        civilian_id = civilian.get_entity_id()
        position_id = civilian.get_position()
        distance_value = area_distance(self._world_info, self._agent_info.get_entity_id(), civilian_id)
        refuge_id, refuge_distance = nearest_refuge(self._world_info, position_id)
        total_trip_distance = distance_value + refuge_distance
        life_margin = estimate_life_margin(civilian, total_trip_distance)
        urgency = urgency_score(civilian)
        competitors = count_ambulances_at(self._world_info, position_id, self._agent_info.get_entity_id())
        in_cluster = position_id in cluster_ids if position_id is not None else False
        same_position = self._agent_info.get_position_entity_id() == position_id and position_id is not None
        current_target = self._result is not None and civilian_id == self._result
        current_target_age = self._current_target_age(runtime_state, tick) if current_target else 0
        blocked = runtime_state.is_blocked('detector', civilian_id, tick)
        survival_ratio = life_margin / max(total_trip_distance / 30000.0, 1.0)
        heuristic_score = 0.0
        heuristic_score += float(setting('detector.weights.life_margin_weight', 550.0)) * life_margin
        heuristic_score += float(setting('detector.weights.urgency_weight', 65.0)) * urgency
        heuristic_score -= float(setting('detector.weights.distance_weight', 0.18)) * distance_value
        heuristic_score -= float(setting('detector.weights.refuge_distance_weight', 0.06)) * refuge_distance
        heuristic_score -= float(setting('detector.weights.competition_penalty', 2500.0)) * competitors

        same_position_bonus = float(setting('detector.weights.same_position_bonus', 3000.0)) if same_position else 0.0
        heuristic_score += same_position_bonus

        current_target_bonus = 0.0
        current_target_penalty = 0.0
        if current_target:
            current_target_bonus = float(setting('detector.weights.current_target_bonus', 4500.0))
            decay_start = int(setting('detector.weights.current_target_bonus_decay_start_ticks', setting('detector.selection.keep_target_ticks', 5)))
            decay_per_tick = float(setting('detector.weights.current_target_bonus_decay_per_tick', 900.0))
            stale_penalty_per_tick = float(setting('detector.weights.current_target_stale_penalty_per_tick', 450.0))
            stale_ticks = max(current_target_age - max(decay_start, 0), 0)
            if stale_ticks > 0:
                current_target_bonus = max(current_target_bonus - decay_per_tick * stale_ticks, 0.0)
                if not same_position:
                    current_target_penalty = stale_penalty_per_tick * stale_ticks
        heuristic_score += current_target_bonus
        heuristic_score -= current_target_penalty

        heuristic_score += float(setting('detector.weights.cluster_bonus', 2000.0)) if in_cluster else 0.0
        heuristic_score -= float(setting('detector.weights.blocked_penalty', 50000.0)) if blocked else 0.0
        return {
            'candidate_id': entity_value(civilian_id),
            'position_id': entity_value(position_id),
            'distance': round(distance_value, 2),
            'hp': civilian.get_hp(),
            'damage': civilian.get_damage(),
            'buriedness': civilian.get_buriedness(),
            'life_margin': round(life_margin, 4),
            'urgency': round(urgency, 4),
            'competitors': competitors,
            'in_cluster': in_cluster,
            'same_position': same_position,
            'current_target': current_target,
            'current_target_age': current_target_age,
            'current_target_bonus_applied': round(current_target_bonus, 4),
            'current_target_penalty_applied': round(current_target_penalty, 4),
            'same_position_bonus_applied': round(same_position_bonus, 4),
            'blocked': blocked,
            'refuge_id': entity_value(refuge_id),
            'refuge_distance': round(refuge_distance, 2),
            'total_trip_distance': round(total_trip_distance, 2),
            'survival_ratio': round(survival_ratio, 4),
            'path_nodes': 0,
            'reachable': False,
            'heuristic_score': round(heuristic_score, 4),
        }

    def _probe_path(self, payload: dict[str, object], by_id: dict[str, EntityID]) -> bool:
        candidate_id = by_id.get(str(payload.get('candidate_id')))
        if candidate_id is None:
            payload['path_nodes'] = 0
            payload['reachable'] = False
            return False
        path = self._path_planning.get_path(self._agent_info.get_entity_id(), candidate_id)
        payload['path_nodes'] = len(path)
        payload['reachable'] = bool(path)
        return bool(path)


    def _phase_for_ml(self, runtime_state) -> str:
        phase = str(getattr(runtime_state, 'phase', '') or '').strip()
        if phase in {'search', 'transport', 'move_to_victim'}:
            return phase
        return 'move_to_victim' if self._result is not None else 'search'

    def _log_ml_runtime_state(self, runtime_state, scope_info: dict[str, object], candidate_payload: list[dict[str, object]], tick: int) -> None:
        if tick > 5:
            return
        bridge_info = self._ml.describe()
        first_candidate = dict(candidate_payload[0]) if candidate_payload else {}
        bridge_info.update(
            {
                'tick': tick,
                'runtime_phase': self._phase_for_ml(runtime_state),
                'deferred_rescue_active': bool(runtime_state.deferred_rescue_active),
                'candidate_scope': scope_info.get('candidate_scope'),
                'scope_selected_count': scope_info.get('scope_selected_count'),
                'candidate_count': len(candidate_payload),
                'current_detector_target': entity_value(self._result),
                'feature_keys': sorted(first_candidate.keys()),
            }
        )
        self._decision_logger.debug_event('detector_ml_runtime_state', bridge_info)

    def _apply_ml_scores(self, context: dict[str, object], candidate_payload: list[dict[str, object]]) -> None:
        ml_scores = self._ml.score_candidates(context, candidate_payload)
        mode = self._ml.mode_name()
        heuristic_weight = float(setting('detector.ml.heuristic_weight', 0.15))
        ml_weight = float(setting('detector.ml.ml_weight', 0.85))

        heuristic_values = [float(item.get('heuristic_score') or 0.0) for item in candidate_payload]
        h_min = min(heuristic_values) if heuristic_values else 0.0
        h_max = max(heuristic_values) if heuristic_values else 0.0
        h_range = max(h_max - h_min, 1e-9)

        for payload in candidate_payload:
            candidate_id = str(payload.get('candidate_id'))
            ml_score = ml_scores.get(candidate_id)
            payload['ml_score'] = round(ml_score, 6) if ml_score is not None else None

            heuristic_score = float(payload.get('heuristic_score') or 0.0)
            if len(candidate_payload) <= 1:
                heuristic_component = 1.0
            else:
                heuristic_component = max(0.0, min(1.0, (heuristic_score - h_min) / h_range))
            payload['heuristic_component'] = round(heuristic_component, 6)

            if mode in {'hybrid', 'pure_ml_test'} and ml_score is not None:
                if mode == 'pure_ml_test':
                    final_score = float(ml_score)
                else:
                    final_score = ml_weight * float(ml_score) + heuristic_weight * heuristic_component
            else:
                final_score = heuristic_component
            payload['ml_component'] = round(float(ml_score), 6) if ml_score is not None else None
            payload['final_score'] = round(final_score, 6)

    def _ranking_diagnostics(self, candidate_payload: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidate_payload:
            return {
                'heuristic_best_id': None,
                'ml_best_id': None,
                'final_best_id': None,
                'winner_changed_by_ml': False,
                'override_applied': False,
                'ml_gap': None,
                'heuristic_gap': None,
            }

        heuristic_sorted = sorted(candidate_payload, key=lambda item: float(item.get('heuristic_score') or 0.0), reverse=True)
        ml_sorted = sorted([item for item in candidate_payload if item.get('ml_score') is not None], key=lambda item: float(item.get('ml_score') or 0.0), reverse=True)
        final_sorted = sorted(candidate_payload, key=lambda item: float(item.get('final_score') or 0.0), reverse=True)

        heuristic_best_id = str(heuristic_sorted[0].get('candidate_id')) if heuristic_sorted else None
        ml_best_id = str(ml_sorted[0].get('candidate_id')) if ml_sorted else None
        final_best_id = str(final_sorted[0].get('candidate_id')) if final_sorted else None

        ml_gap = None
        if len(ml_sorted) >= 2:
            ml_gap = round(float(ml_sorted[0].get('ml_score') or 0.0) - float(ml_sorted[1].get('ml_score') or 0.0), 6)

        heuristic_gap = None
        if len(heuristic_sorted) >= 2:
            heuristic_gap = round(float(heuristic_sorted[0].get('heuristic_component') or 0.0) - float(heuristic_sorted[1].get('heuristic_component') or 0.0), 6)

        return {
            'heuristic_best_id': heuristic_best_id,
            'ml_best_id': ml_best_id,
            'final_best_id': final_best_id,
            'winner_changed_by_ml': bool(final_best_id is not None and heuristic_best_id is not None and final_best_id != heuristic_best_id),
            'override_applied': False,
            'ml_gap': ml_gap,
            'heuristic_gap': heuristic_gap,
        }

    def _selection_order(self, candidate_payload: list[dict[str, Any]], diagnostics: dict[str, Any], entity_by_id: dict[str, EntityID], runtime_state, tick: int) -> tuple[list[dict[str, Any]], str, bool, str, dict[str, Any]]:
        ordered_by_heuristic = sorted(candidate_payload, key=lambda item: float(item.get('heuristic_score') or 0.0), reverse=True)
        mode = self._ml.mode_name()
        diagnostics = dict(diagnostics)
        strict_threshold = float(setting('detector.ml.override_min_delta', 0.03))
        soft_threshold = float(setting('detector.ml.override_soft_min_delta', max(strict_threshold * 0.5, 0.015)))
        strict_confidence = float(setting('detector.ml.min_confidence', 0.0))
        soft_confidence = float(setting('detector.ml.override_soft_min_confidence', max(0.0, strict_confidence - 0.07)))
        strict_margin = float(setting('detector.ml.min_margin', 0.0))
        soft_margin = float(setting('detector.ml.override_soft_min_margin', strict_margin))
        diagnostics.setdefault('override_delta', None)
        diagnostics.setdefault('override_threshold', strict_threshold)
        diagnostics.setdefault('override_soft_threshold', soft_threshold)
        diagnostics.setdefault('override_ml_reachable', None)
        diagnostics.setdefault('override_heuristic_reachable', None)
        diagnostics.setdefault('override_blocked_reason', None)
        diagnostics.setdefault('override_max_trip_ratio', float(setting('detector.ml.override_max_trip_ratio', 1.2)))
        diagnostics.setdefault('override_soft_max_trip_ratio', float(setting('detector.ml.override_soft_max_trip_ratio', 1.35)))
        diagnostics.setdefault('override_min_life_margin', float(setting('detector.ml.override_min_life_margin', 20.0)))
        diagnostics.setdefault('override_trip_advantage_ratio', float(setting('detector.ml.override_trip_advantage_ratio', 0.92)))
        diagnostics.setdefault('override_life_margin_advantage', float(setting('detector.ml.override_life_margin_advantage', 15.0)))
        diagnostics.setdefault('override_current_target_stale_ticks', int(setting('detector.ml.override_current_target_stale_ticks', 4)))
        diagnostics.setdefault('override_tier', 'none')
        diagnostics.setdefault('override_contexts', [])

        if mode not in {'hybrid', 'pure_ml_test'}:
            diagnostics['final_best_id'] = str(ordered_by_heuristic[0].get('candidate_id')) if ordered_by_heuristic else None
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'ml_inactive'
            return ordered_by_heuristic, 'heuristic', False, 'heuristic_keep_victim', diagnostics

        ml_sorted = sorted([item for item in candidate_payload if item.get('ml_score') is not None], key=lambda item: float(item.get('ml_score') or 0.0), reverse=True)
        if not ml_sorted:
            diagnostics['final_best_id'] = str(ordered_by_heuristic[0].get('candidate_id')) if ordered_by_heuristic else None
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'no_ml_scores'
            return ordered_by_heuristic, 'heuristic_fallback', False, 'heuristic_keep_victim', diagnostics

        heuristic_best = ordered_by_heuristic[0] if ordered_by_heuristic else None
        ml_best = ml_sorted[0] if ml_sorted else None
        if heuristic_best is None or ml_best is None:
            diagnostics['final_best_id'] = str(ordered_by_heuristic[0].get('candidate_id')) if ordered_by_heuristic else None
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'missing_best_candidate'
            return ordered_by_heuristic, 'heuristic_fallback', False, 'heuristic_keep_victim', diagnostics

        top1 = float(ml_best.get('ml_score') or 0.0)
        top2 = float(ml_sorted[1].get('ml_score') or 0.0) if len(ml_sorted) >= 2 else 0.0
        ml_margin = top1 - top2
        heuristic_best_id = str(heuristic_best.get('candidate_id'))
        ml_best_id = str(ml_best.get('candidate_id'))
        if heuristic_best_id == ml_best_id:
            diagnostics['final_best_id'] = heuristic_best_id
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'same_winner'
            return ordered_by_heuristic, 'heuristic', False, 'heuristic_keep_victim', diagnostics

        heuristic_ml = heuristic_best.get('ml_score')
        if heuristic_ml is None:
            diagnostics['final_best_id'] = heuristic_best_id
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'missing_heuristic_ml_score'
            return ordered_by_heuristic, 'heuristic_fallback', False, 'heuristic_keep_victim', diagnostics

        heuristic_reachable = self._probe_path(heuristic_best, entity_by_id)
        ml_reachable = self._probe_path(ml_best, entity_by_id)
        diagnostics['override_heuristic_reachable'] = heuristic_reachable
        diagnostics['override_ml_reachable'] = ml_reachable

        heuristic_trip = float(heuristic_best.get('total_trip_distance') or 0.0)
        ml_trip = float(ml_best.get('total_trip_distance') or 0.0)
        heuristic_life = float(heuristic_best.get('life_margin') or 0.0)
        ml_life = float(ml_best.get('life_margin') or 0.0)
        same_position_context = bool(
            heuristic_best.get('position_id') is not None
            and heuristic_best.get('position_id') == ml_best.get('position_id')
        )
        ml_best_same_position = bool(ml_best.get('same_position')) and not bool(heuristic_best.get('same_position'))
        trip_advantage = bool(heuristic_trip > 0.0 and ml_trip <= heuristic_trip * float(diagnostics['override_trip_advantage_ratio']))
        life_advantage = bool(ml_life >= heuristic_life + float(diagnostics['override_life_margin_advantage']))
        stale_current_target = bool(
            heuristic_best.get('current_target')
            and self._current_target_age(runtime_state, tick) >= int(diagnostics['override_current_target_stale_ticks'])
        )
        near_tie_context = bool(
            diagnostics.get('heuristic_gap') is not None
            and float(diagnostics.get('heuristic_gap') or 0.0) <= float(setting('detector.ml.override_near_tie_gap', 0.03))
        )
        relaxed_contexts: list[str] = []
        if same_position_context:
            relaxed_contexts.append('same_position')
        if ml_best_same_position:
            relaxed_contexts.append('ml_best_same_position')
        if near_tie_context:
            relaxed_contexts.append('near_tie')
        if trip_advantage:
            relaxed_contexts.append('trip_advantage')
        if life_advantage:
            relaxed_contexts.append('life_advantage')
        if stale_current_target:
            relaxed_contexts.append('stale_current_target')
        if not heuristic_reachable:
            relaxed_contexts.append('heuristic_unreachable')
        if bool(heuristic_best.get('blocked')):
            relaxed_contexts.append('heuristic_blocked')
        diagnostics['override_contexts'] = relaxed_contexts

        strict_conf_ok = top1 >= strict_confidence and (len(ml_sorted) == 1 or ml_margin >= strict_margin)
        soft_conf_ok = top1 >= soft_confidence and (len(ml_sorted) == 1 or ml_margin >= soft_margin)
        if not (strict_conf_ok or (soft_conf_ok and relaxed_contexts)):
            diagnostics['final_best_id'] = heuristic_best_id
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'low_confidence'
            return ordered_by_heuristic, 'heuristic_fallback', False, 'heuristic_keep_victim', diagnostics

        if not ml_reachable:
            diagnostics['final_best_id'] = heuristic_best_id
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'ml_unreachable'
            return ordered_by_heuristic, 'heuristic', False, 'heuristic_keep_victim', diagnostics

        override_delta = float(ml_best.get('ml_score') or 0.0) - float(heuristic_ml)
        diagnostics['override_delta'] = round(override_delta, 6)
        threshold_used = strict_threshold
        max_trip_ratio = float(diagnostics['override_max_trip_ratio'])
        override_tier = 'strict'
        if relaxed_contexts and soft_conf_ok:
            threshold_used = min(strict_threshold, soft_threshold)
            max_trip_ratio = max(float(diagnostics['override_max_trip_ratio']), float(diagnostics['override_soft_max_trip_ratio']))
            if same_position_context or ml_best_same_position:
                max_trip_ratio = max(max_trip_ratio, float(setting('detector.ml.override_same_position_max_trip_ratio', 1.5)))
            override_tier = 'relaxed'
        diagnostics['override_threshold'] = threshold_used
        diagnostics['override_tier'] = override_tier
        if override_delta < threshold_used:
            diagnostics['final_best_id'] = heuristic_best_id
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'delta_below_threshold'
            return ordered_by_heuristic, 'heuristic', False, 'heuristic_keep_victim', diagnostics

        if heuristic_reachable and heuristic_trip > 0.0 and ml_trip > heuristic_trip * max_trip_ratio:
            diagnostics['final_best_id'] = heuristic_best_id
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'trip_ratio_too_high'
            return ordered_by_heuristic, 'heuristic', False, 'heuristic_keep_victim', diagnostics

        min_life_margin = float(diagnostics['override_min_life_margin'])
        if relaxed_contexts and (same_position_context or ml_best_same_position or near_tie_context):
            min_life_margin = min(min_life_margin, float(setting('detector.ml.override_soft_min_life_margin', 8.0)))
        if ml_life < min_life_margin and not same_position_context and not ml_best_same_position:
            diagnostics['final_best_id'] = heuristic_best_id
            diagnostics['winner_changed_by_ml'] = False
            diagnostics['override_applied'] = False
            diagnostics['override_blocked_reason'] = 'life_margin_too_low'
            return ordered_by_heuristic, 'heuristic', False, 'heuristic_keep_victim', diagnostics

        ordered = [ml_best] + [item for item in ordered_by_heuristic if item is not ml_best]
        diagnostics['final_best_id'] = ml_best_id
        diagnostics['winner_changed_by_ml'] = bool(heuristic_best_id != ml_best_id)
        diagnostics['override_applied'] = True
        diagnostics['override_blocked_reason'] = None
        return ordered, 'ml_override', False, 'ml_override_victim', diagnostics
    def _nearby_unvisited_buildings(self, source_building_id: Optional[EntityID], runtime_state) -> list[Building]:
        if source_building_id is None:
            return []
        radius = float(setting('search.deferred_rescue.search_radius', 70000.0))
        current_area = self._agent_info.get_position_entity_id()
        buildings: list[Building] = []
        for building in searchable_buildings(self._world_info):
            building_id = building.get_entity_id()
            if building_id is None or building_id == source_building_id:
                continue
            key = entity_value(building_id)
            entered = getattr(runtime_state, 'entered_buildings', None)
            if isinstance(entered, set) and key in entered:
                continue
            if not isinstance(entered, set) and key in runtime_state.visited_buildings:
                continue
            if runtime_state.is_blocked('search', building_id, int(self._agent_info.get_time())):
                continue
            if area_distance(self._world_info, source_building_id, building_id) > radius:
                continue
            if current_area is not None and area_distance(self._world_info, current_area, building_id) > radius * 1.35:
                continue
            buildings.append(building)
        buildings.sort(key=lambda b: area_distance(self._world_info, current_area, b.get_entity_id()))
        limit = int(setting('detector.defer.max_nearby_unvisited_buildings', 4))
        return buildings[:max(0, limit)]

    def _scoped_candidates(self, candidates: list[Civilian], runtime_state, tick: int) -> tuple[list[Civilian], dict[str, object]]:
        if not candidates:
            return [], {
                'candidate_scope': 'empty',
                'scope_radius': float(setting('detector.scope.local_radius', 120000.0)),
                'scope_world_count': 0,
                'scope_local_count': 0,
                'scope_selected_count': 0,
            }
        local_radius = float(setting('detector.scope.local_radius', 120000.0))
        max_local = max(1, int(setting('detector.scope.max_local_candidates', 12)))
        max_global = max(max_local, int(setting('detector.scope.max_global_candidates', 18)))
        min_local = max(1, int(setting('detector.scope.min_local_candidates_to_lock', 1)))
        keep_current = bool(setting('detector.scope.always_include_current_target', True))
        keep_same_building = bool(setting('detector.scope.always_include_same_building_victims', True))
        current_area = self._agent_info.get_position_entity_id()
        current_target_id = self._result

        scored: list[tuple[Civilian, float, bool, bool]] = []
        for civilian in candidates:
            civilian_id = civilian.get_entity_id()
            if civilian_id is None:
                continue
            distance_value = area_distance(self._world_info, self._agent_info.get_entity_id(), civilian_id)
            same_building = current_area is not None and civilian.get_position() == current_area
            is_current_target = current_target_id is not None and civilian_id == current_target_id
            scored.append((civilian, distance_value, same_building, is_current_target))
        scored.sort(key=lambda item: (0 if item[2] else 1, item[1]))

        local = [item for item in scored if item[2] or item[1] <= local_radius]
        selected_bucket = local if len(local) >= min_local else scored
        scope_name = 'local_radius' if len(local) >= min_local else 'nearest_global_fallback'
        limit = max_local if scope_name == 'local_radius' else max_global
        selected = selected_bucket[:limit]
        present = {item[0].get_entity_id() for item in selected if item[0].get_entity_id() is not None}

        global_challenger_count = max(0, int(setting('detector.scope.global_challenger_count', 0)))
        alternate_challenger_count = max(0, int(setting('detector.scope.alternate_challenger_count', max(global_challenger_count, 2))))
        scope_global_challengers = 0
        scope_alternate_challengers = 0
        if scope_name == 'local_radius' and global_challenger_count > 0:
            for item in scored:
                civilian = item[0]
                civilian_id = civilian.get_entity_id()
                if civilian_id is None or civilian_id in present:
                    continue
                if len(selected) >= max_global:
                    break
                selected.append(item)
                present.add(civilian_id)
                scope_global_challengers += 1
                if scope_global_challengers >= global_challenger_count:
                    break

        if alternate_challenger_count > 0 and len(selected) < max_global:
            alternates = sorted(
                [item for item in scored if item[0].get_entity_id() is not None and item[0].get_entity_id() not in present],
                key=lambda item: (
                    -(estimate_life_margin(item[0], item[1] + nearest_refuge(self._world_info, item[0].get_position())[1])),
                    -urgency_score(item[0]),
                    item[1],
                ),
            )
            for item in alternates:
                civilian_id = item[0].get_entity_id()
                if civilian_id is None or civilian_id in present:
                    continue
                if len(selected) >= max_global:
                    break
                selected.append(item)
                present.add(civilian_id)
                scope_alternate_challengers += 1
                if scope_alternate_challengers >= alternate_challenger_count:
                    break

        if keep_same_building and current_area is not None:
            for item in scored:
                civilian = item[0]
                civilian_id = civilian.get_entity_id()
                if civilian_id is None:
                    continue
                if civilian.get_position() == current_area and civilian_id not in present:
                    selected.append(item)
                    present.add(civilian_id)

        if keep_current and current_target_id is not None:
            if current_target_id not in present:
                for item in scored:
                    civilian_id = item[0].get_entity_id()
                    if civilian_id == current_target_id:
                        selected.append(item)
                        present.add(civilian_id)
                        break

        selected.sort(key=lambda item: (0 if item[2] else 1, item[1]))
        scoped = [item[0] for item in selected]
        info = {
            'candidate_scope': scope_name,
            'scope_radius': local_radius,
            'scope_world_count': len(scored),
            'scope_local_count': len(local),
            'scope_selected_count': len(scoped),
            'scope_current_area': entity_value(current_area),
            'scope_global_challengers': scope_global_challengers,
            'scope_alternate_challengers': scope_alternate_challengers,
        }
        return scoped, info

    def _should_defer_rescue(
        self,
        chosen: Optional[EntityID],
        chosen_payload: Optional[dict[str, object]],
        candidate_payload: list[dict[str, object]],
        runtime_state,
        tick: int,
    ) -> tuple[bool, str, dict[str, object]]:
        if runtime_state.deferred_rescue_active:
            runtime_state.stop_deferred_rescue(tick)
        return False, 'disabled', {}

    def _select_candidate_entity(self, candidate_payload: list[dict[str, object]], entity_by_id: dict[str, EntityID], runtime_state, tick: int, selection_order: list[dict[str, object]]) -> tuple[Optional[EntityID], str]:
        top_k = max(1, int(setting('detector.selection.path_check_top_k', 4)))
        chosen: Optional[EntityID] = None
        chosen_reason = 'highest_score'
        for payload in selection_order[:max(top_k, len(selection_order))]:
            reachable = bool(payload.get('reachable'))
            if not reachable:
                reachable = self._probe_path(payload, entity_by_id)
            if reachable:
                candidate_key = str(payload.get('candidate_id'))
                chosen = entity_by_id.get(candidate_key)
                break
            candidate_key = str(payload.get('candidate_id'))
            if candidate_key in entity_by_id:
                runtime_state.block_target('detector', entity_by_id[candidate_key], tick + int(setting('detector.selection.unreachable_block_ticks', 18)))
        if chosen is None:
            chosen_reason = 'no_reachable_candidate'
        return chosen, chosen_reason

    def calculate(self) -> HumanDetector:
        started = time.monotonic()
        tick = int(self._agent_info.get_time())
        runtime_state = get_runtime_state(self._agent_info)
        if tick <= 1:
            self._decision_logger.debug_event('ml_v2_stage2_fix_active', {'module': 'detector', 'fix_version': 'stable_release'})
        runtime_state.cleanup(tick)
        runtime_state.update_position(self._agent_info.get_position_entity_id(), tick)
        carried: Optional[Human] = self._agent_info.some_one_on_board()
        current_area = self._agent_info.get_position_entity_id()

        if runtime_state.deferred_rescue_active:
            runtime_state.stop_deferred_rescue(tick)


        if carried is not None:
            self._result = carried.get_entity_id()
            runtime_state.carrying = True
            runtime_state.phase = 'transport'
            runtime_state.set_detector_target(self._result, tick)
            refuge_id, refuge_distance = nearest_refuge(self._world_info, self._agent_info.get_position_entity_id())
            runtime_state.set_refuge_target(refuge_id, tick)
            if runtime_state.deferred_rescue_active:
                runtime_state.stop_deferred_rescue(tick)
            self._decision_logger.log_text(
                'Агент уже несет пострадавшего',
                {
                    'цель': entity_value(self._result),
                    'убежище': entity_value(refuge_id),
                    'дистанция_до_убежища': round(refuge_distance, 2),
                    'режим_детектора': self._ml.mode_name(),
                },
            )
            self._decision_logger.decision_snapshot(
                'detector',
                {
                    'current_area': entity_value(self._agent_info.get_position_entity_id()),
                    'mode': 'carrying',
                    'phase': runtime_state.phase,
                    'refuge_target': entity_value(refuge_id),
                    'detector_mode': self._ml.mode_name(),
                    'detector_requested_mode': self._ml.requested_mode,
                    'compute_ms': round((time.monotonic() - started) * 1000.0, 3),
                },
                [],
                entity_value(self._result),
                'carrying',
            )
            self._decision_logger.state_snapshot('agent_runtime_state', runtime_state.snapshot())
            return self

        runtime_state.carrying = False
        me = self._agent_info.get_entity_id()
        cluster_entities = cluster_entities_for_agent(self._clustering, me)
        cluster_ids = {entity.get_entity_id() for entity in cluster_entities if getattr(entity, 'get_entity_id', None)}
        cluster_targets = [entity for entity in cluster_entities if isinstance(entity, Civilian) and is_rescuable_civilian(self._world_info, entity)]
        world_targets = [entity for entity in self._world_info.get_entities_of_types([Civilian]) if isinstance(entity, Civilian) and is_rescuable_civilian(self._world_info, entity)]
        all_candidates = [entity for entity in world_targets if entity.get_entity_id() is not None]

        if not all_candidates:
            self._result = None
            runtime_state.set_detector_target(None, tick)
            runtime_state.phase = 'search'
            if runtime_state.deferred_rescue_active:
                runtime_state.stop_deferred_rescue(tick)
            self._decision_logger.log_text('Подходящих пострадавших для спасения нет', {'фаза': runtime_state.phase, 'режим_детектора': self._ml.mode_name()})
            empty_metadata = {
                'selection_mode': self._ml.mode_name(),
                'selected_by': 'heuristic',
                'exploration_used': False,
                'selected_reason': 'no_candidates',
                'top_k_candidates': [],
            }
            self._decision_logger.decision_snapshot(
                'detector',
                {
                    'current_area': entity_value(self._agent_info.get_position_entity_id()),
                    'cluster_candidate_count': len(cluster_targets),
                    'global_candidate_count': len(world_targets),
                    'scoped_candidate_count': 0,
                    'candidate_scope': 'empty',
                    'phase': runtime_state.phase,
                    'known_refuges': len(refuge_entities(self._world_info)),
                    'known_civilians': len(visible_civilians(self._world_info)),
                    'detector_mode': self._ml.mode_name(),
                    'detector_requested_mode': self._ml.requested_mode,
                    'compute_ms': round((time.monotonic() - started) * 1000.0, 3),
                },
                [],
                None,
                'no_candidates',
                metadata=empty_metadata,
            )
            self._decision_logger.state_snapshot('agent_runtime_state', runtime_state.snapshot())
            return self

        candidates, scope_info = self._scoped_candidates(all_candidates, runtime_state, tick)
        candidate_payload = [self._candidate_features(civilian, cluster_ids, runtime_state, tick) for civilian in candidates]
        candidate_payload.sort(key=lambda item: float(item.get('heuristic_score') or 0.0), reverse=True)
        for index, payload in enumerate(candidate_payload, start=1):
            payload['heuristic_rank'] = index

        probe_top_k = max(1, int(setting('detector.selection.feature_path_probe_top_k', setting('detector.selection.path_check_top_k', 4))))
        entity_by_id: dict[str, EntityID] = {}
        for civilian in candidates:
            candidate_id = civilian.get_entity_id()
            if candidate_id is not None:
                entity_by_id[entity_value(candidate_id)] = candidate_id
        for payload in candidate_payload[:probe_top_k]:
            self._probe_path(payload, entity_by_id)

        context = {
            'tick': tick,
            'current_area': entity_value(current_area),
            'candidate_count': len(candidate_payload),
            'phase': self._phase_for_ml(runtime_state),
            **scope_info,
            'known_refuges': len(refuge_entities(self._world_info)),
            'known_civilians': len(visible_civilians(self._world_info)),
            'current_target_age': self._current_target_age(runtime_state, tick),
            'active_search_target': entity_value(runtime_state.search_target),
            'cluster_candidate_count': len(cluster_targets),
            'global_candidate_count': len(all_candidates),
            'scoped_candidate_count': len(candidate_payload),
            'deferred_rescue_active': bool(runtime_state.deferred_rescue_active),
        }
        self._log_ml_runtime_state(runtime_state, scope_info, candidate_payload, tick)
        self._apply_ml_scores(context, candidate_payload)
        ml_ranked = [item for item in candidate_payload if item.get('ml_score') is not None]
        ml_ranked.sort(key=lambda item: float(item.get('ml_score') or 0.0), reverse=True)
        for index, payload in enumerate(ml_ranked, start=1):
            payload['ml_rank'] = index
        candidate_payload.sort(key=lambda item: float(item.get('final_score') or 0.0), reverse=True)
        for index, payload in enumerate(candidate_payload, start=1):
            payload['final_rank'] = index

        ranking_diagnostics = self._ranking_diagnostics(candidate_payload)
        selection_order, selected_by, exploration_used, preferred_reason, ranking_diagnostics = self._selection_order(candidate_payload, ranking_diagnostics, entity_by_id, runtime_state, tick)
        chosen, chosen_reason = self._select_candidate_entity(candidate_payload, entity_by_id, runtime_state, tick, selection_order)
        if chosen is not None and chosen_reason == 'highest_score':
            chosen_reason = preferred_reason

        keep_target_ticks = int(setting('detector.selection.keep_target_ticks', 6))
        sticky_tolerance = float(setting('detector.selection.sticky_keep_ml_tolerance', 0.03))
        sticky_break_delta = float(setting('detector.selection.sticky_keep_break_delta', max(sticky_tolerance, 0.02)))
        sticky_max_rank = max(1, int(setting('detector.selection.sticky_keep_max_heuristic_rank', 2)))
        current_best = selection_order[0] if selection_order else None
        current_payload = next((item for item in candidate_payload if bool(item.get('current_target'))), None)
        if selected_by != 'ml_override' and not exploration_used and self._result is not None and current_payload is not None and current_best is not None:
            current_age = self._current_target_age(runtime_state, tick)
            best_ml = float(current_best.get('ml_score') or 0.0)
            current_ml = float(current_payload.get('ml_score') or 0.0)
            current_reachable = bool(current_payload.get('reachable'))
            if not current_reachable:
                current_reachable = self._probe_path(current_payload, entity_by_id)
            current_rank = int(current_payload.get('heuristic_rank') or 999)
            challenger_breaks = False
            if current_best is not current_payload and current_best.get('candidate_id') != current_payload.get('candidate_id'):
                challenger_ml = float(current_best.get('ml_score') or 0.0)
                challenger_trip = float(current_best.get('total_trip_distance') or 0.0)
                current_trip = float(current_payload.get('total_trip_distance') or 0.0)
                challenger_life = float(current_best.get('life_margin') or 0.0)
                current_life = float(current_payload.get('life_margin') or 0.0)
                challenger_same_position = bool(
                    current_best.get('position_id') is not None
                    and current_best.get('position_id') == current_payload.get('position_id')
                )
                challenger_breaks = (
                    challenger_ml - current_ml >= sticky_break_delta
                    and (
                        challenger_same_position
                        or (current_trip > 0.0 and challenger_trip <= current_trip * float(setting('detector.ml.override_soft_max_trip_ratio', 1.35)))
                        or challenger_life >= current_life + float(setting('detector.ml.override_life_margin_advantage', 15.0))
                    )
                )
            if current_reachable and current_age <= keep_target_ticks and best_ml - current_ml <= sticky_tolerance and current_rank <= sticky_max_rank and not challenger_breaks:
                chosen = self._result
                chosen_reason = 'sticky_keep'

        chosen_payload = next((item for item in candidate_payload if item.get('candidate_id') == entity_value(chosen)), None)

        if chosen is not None:
            ranking_diagnostics['final_best_id'] = entity_value(chosen)
            ranking_diagnostics['winner_changed_by_ml'] = bool(ranking_diagnostics.get('heuristic_best_id') and ranking_diagnostics.get('final_best_id') and ranking_diagnostics.get('heuristic_best_id') != ranking_diagnostics.get('final_best_id'))
            if selected_by in {'ml_primary', 'ml_exploration'} and ranking_diagnostics.get('heuristic_best_id') != entity_value(chosen):
                ranking_diagnostics['override_applied'] = True

        selection_metadata = self._selection_metadata(candidate_payload, chosen, self._ml.mode_name(), chosen_reason, exploration_used=exploration_used)
        selection_metadata.update(ranking_diagnostics)
        selection_metadata['selected_by'] = selected_by
        self._result = chosen
        runtime_state.set_detector_target(chosen, tick)
        runtime_state.register_detector_selection(
            chosen,
            tick,
            int(setting('detector.ml_v2.outcome_window_ticks', 8)),
            selected_rank=int(selection_metadata.get('selected_rank_by_heuristic') or -1),
            ml_rank=int(selection_metadata.get('selected_rank_by_ml') or -1) if selection_metadata.get('selected_rank_by_ml') is not None else -1,
            final_rank=int(selection_metadata.get('selected_rank_by_final') or -1),
            selected_by=str(selection_metadata.get('selected_by') or 'heuristic'),
            selection_mode=str(selection_metadata.get('selection_mode') or self._ml.mode_name()),
            reason=str(selection_metadata.get('selected_reason') or chosen_reason),
            exploration_used=bool(selection_metadata.get('exploration_used', False)),
            candidate_count=len(candidate_payload),
            top_k_candidates=list(selection_metadata.get('top_k_candidates') or []),
        )
        refuge_id = None
        refuge_distance = float('inf')
        if chosen is not None:
            chosen_entity = self._world_info.get_entity(chosen)
            if isinstance(chosen_entity, Civilian):
                refuge_id, refuge_distance = nearest_refuge(self._world_info, chosen_entity.get_position())
                runtime_state.set_refuge_target(refuge_id, tick)
            runtime_state.phase = 'move_to_victim'
        else:
            runtime_state.set_refuge_target(None, tick)
            runtime_state.phase = 'search'

        state = {
            'current_area': entity_value(self._agent_info.get_position_entity_id()),
            'cluster_candidate_count': len(cluster_targets),
            'global_candidate_count': len(world_targets),
            'scoped_candidate_count': len(candidate_payload),
            'candidate_scope': scope_info.get('candidate_scope'),
            'scope_world_count': scope_info.get('scope_world_count'),
            'scope_local_count': scope_info.get('scope_local_count'),
            'used_cluster_only': False,
            'phase': runtime_state.phase,
            'known_refuges': len(refuge_entities(self._world_info)),
            'known_civilians': len(visible_civilians(self._world_info)),
            'active_search_target': entity_value(runtime_state.search_target),
            'detector_mode': self._ml.mode_name(),
            'detector_requested_mode': self._ml.requested_mode,
            'deferred_rescue_active': False,
            'heuristic_best_id': selection_metadata.get('heuristic_best_id'),
            'ml_best_id': selection_metadata.get('ml_best_id'),
            'final_best_id': selection_metadata.get('final_best_id'),
            'winner_changed_by_ml': selection_metadata.get('winner_changed_by_ml'),
            'override_applied': selection_metadata.get('override_applied'),
            'ml_gap': selection_metadata.get('ml_gap'),
            'heuristic_gap': selection_metadata.get('heuristic_gap'),
        }
        self._decision_logger.decision_snapshot('detector', state, candidate_payload, entity_value(chosen) if chosen is not None else None, chosen_reason, metadata=selection_metadata)
        top_preview = '; '.join(
            f"{item.get('candidate_id')}:{item.get('final_score')}|ml={item.get('ml_score')}|reach={item.get('reachable')}"
            for item in candidate_payload[:5]
        )
        self._decision_logger.log_text(
            'Выбрана жертва для спасения',
            {
                'цель': entity_value(chosen),
                'кандидатов': len(candidate_payload),
                'глобальных_кандидатов': len(all_candidates),
                'область_отбора': scope_info.get('candidate_scope'),
                'радиус_отбора': scope_info.get('scope_radius'),
                'причина': chosen_reason,
                'фаза': runtime_state.phase,
                'убежище': entity_value(refuge_id),
                'дистанция_до_убежища': round(refuge_distance, 2) if refuge_distance < float('inf') else 'inf',
                'активная_цель_поиска': entity_value(runtime_state.search_target),
                'режим_детектора': self._ml.mode_name(),
                'запрошенный_режим': self._ml.requested_mode,
                'топ': top_preview,
            },
        )
        self._decision_logger.state_snapshot('agent_runtime_state', runtime_state.snapshot())
        self._decision_logger.debug_event(
            'detector_cycle_summary',
            {
                'compute_ms': round((time.monotonic() - started) * 1000.0, 3),
                'chosen': entity_value(chosen),
                'chosen_reason': chosen_reason,
                'candidate_count': len(candidate_payload),
                'stationary_ticks': runtime_state.stationary_ticks,
                'blocked_first_hops_from_current': runtime_state.blocked_first_hops_from(self._agent_info.get_position_entity_id(), tick),
                'selection_mode': selection_metadata.get('selection_mode'),
                'selected_by': selection_metadata.get('selected_by'),
                'exploration_used': selection_metadata.get('exploration_used'),
                'top_k_candidates': selection_metadata.get('top_k_candidates'),
                'heuristic_best_id': selection_metadata.get('heuristic_best_id'),
                'ml_best_id': selection_metadata.get('ml_best_id'),
                'final_best_id': selection_metadata.get('final_best_id'),
                'winner_changed_by_ml': selection_metadata.get('winner_changed_by_ml'),
                'override_applied': selection_metadata.get('override_applied'),
                'ml_gap': selection_metadata.get('ml_gap'),
                'heuristic_gap': selection_metadata.get('heuristic_gap'),
            },
        )
        return self

    def _selection_source(self, mode: str, exploration_used: bool) -> str:
        if exploration_used:
            return 'exploratory'
        if mode == 'pure_ml_test':
            return 'ml'
        if mode == 'hybrid':
            return 'hybrid'
        return 'heuristic'

    def _selection_metadata(self, candidate_payload: list[dict[str, Any]], chosen: Optional[EntityID], mode: str, selected_reason: str, *, exploration_used: bool = False) -> dict[str, Any]:
        shortlist_k = int(setting('detector.ml_v2.shortlist_k', 5))
        top_k_candidates = [str(item.get('candidate_id')) for item in candidate_payload[:shortlist_k]]
        chosen_key = entity_value(chosen)
        selected_payload = next((item for item in candidate_payload if str(item.get('candidate_id')) == chosen_key), None)
        return {
            'selection_mode': mode,
            'selected_by': self._selection_source(mode, exploration_used),
            'exploration_used': exploration_used,
            'selected_reason': selected_reason,
            'top_k_candidates': top_k_candidates,
            'selected_rank_by_heuristic': int(selected_payload.get('heuristic_rank', -1)) if selected_payload is not None else None,
            'selected_rank_by_ml': int(selected_payload.get('ml_rank', -1)) if selected_payload is not None and selected_payload.get('ml_rank') is not None else None,
            'selected_rank_by_final': int(selected_payload.get('final_rank', -1)) if selected_payload is not None else None,
            'selected_ml_score': round(float(selected_payload.get('ml_score') or 0.0), 6) if selected_payload is not None and selected_payload.get('ml_score') is not None else None,
            'selected_heuristic_score': round(float(selected_payload.get('heuristic_score') or 0.0), 6) if selected_payload is not None else None,
            'selected_final_score': round(float(selected_payload.get('final_score') or 0.0), 6) if selected_payload is not None else None,
        }

    def get_target_entity_id(self) -> Optional[EntityID]:
        return self._result
