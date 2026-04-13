from __future__ import annotations

import heapq
import inspect
import time
from collections import OrderedDict
from itertools import count
from typing import Optional

from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from rcrscore.entities import Building, EntityID, Road

from ..util.decision_logger import DecisionLogger, entity_value
from ..util.ml_bridge import MlBridge
from ..util.rescue_support import area_distance, build_area_graph, get_area_id, path_distance, reconstruct_path, summarize_path
from ..util.runtime_settings import setting
from ..util.shared_runtime_state import get_runtime_state


PATH_LOGIC_VERSION = 'ml_path_planning'


class LoggedAStarPathPlanning(PathPlanning):
    def __init__(
        self,
        agent_info: AgentInfo,
        world_info: WorldInfo,
        scenario_info: ScenarioInfo,
        module_manager: ModuleManager,
        develop_data: DevelopData,
    ) -> None:
        super().__init__(agent_info, world_info, scenario_info, module_manager, develop_data)
        self._graph = build_area_graph(world_info)
        self._decision_logger = DecisionLogger(agent_info, world_info, self.__class__.__name__, 'path')
        self._ml = MlBridge('path')
        self._cache: OrderedDict[tuple[str, str], tuple[int, list[EntityID], dict[str, object]]] = OrderedDict()
        try:
            ml_info = self._ml.describe()
        except Exception:
            ml_info = {'module_type': 'path', 'active_mode': 'describe_failed'}
        self._decision_logger.debug_event('path_ml_module_info', {'path_logic_version': PATH_LOGIC_VERSION, 'ml': ml_info})
        self._decision_logger.log_text('Инициализация path ML', {'logic_version': PATH_LOGIC_VERSION, 'path_ml_mode': ml_info.get('active_mode'), 'model_path': ml_info.get('resolved_model_path')})

    def update_info(self, message_manager: MessageManager) -> PathPlanning:
        super().update_info(message_manager)
        return self

    def calculate(self) -> PathPlanning:
        return self

    def _normalize(self, entity_id: EntityID) -> Optional[EntityID]:
        return get_area_id(self._world_info, entity_id)

    def _tick(self) -> int:
        try:
            return int(self._agent_info.get_time())
        except Exception:
            return 0

    def _cache_key(self, start: Optional[EntityID], goal: Optional[EntityID]) -> tuple[str, str]:
        return entity_value(start), entity_value(goal)

    def _caller_context(self) -> str:
        try:
            frame = inspect.currentframe()
            if frame is None:
                return 'unknown'
            frame = frame.f_back
            depth = 0
            while frame is not None and depth < 12:
                filename = str(frame.f_code.co_filename or '').lower()
                function_name = str(frame.f_code.co_name or '').lower()
                module_name = str(frame.f_globals.get('__name__', '')).lower()
                joined = f'{filename}::{function_name}::{module_name}'
                if 'strategic_search' in joined:
                    return 'search'
                if 'priority_human_detector' in joined:
                    return 'detector'
                if 'default_extend_action_transport' in joined:
                    return 'action_transport'
                if 'default_extend_action_move' in joined:
                    return 'action_move'
                if 'default_extend_action_clear' in joined:
                    return 'action_clear'
                if 'default_command_executor' in joined:
                    return 'command_executor'
                frame = frame.f_back
                depth += 1
            return 'unknown'
        except Exception:
            return 'unknown'

    def _is_action_context(self, caller_context: str) -> bool:
        return caller_context in {'action_move', 'action_transport', 'action_clear', 'command_executor'}

    def _edge_key(self, start: Optional[EntityID], neighbor: Optional[EntityID]) -> str:
        return f'{entity_value(start)}->{entity_value(neighbor)}'

    def _estimate_move_grace_ticks(self, start: Optional[EntityID], path: list[EntityID]) -> int:
        base_ticks = max(1, int(setting('path.recovery.min_attempt_age_ticks_before_fail', 3)))
        max_ticks = max(base_ticks, int(setting('path.recovery.max_attempt_age_ticks_before_fail', 6)))
        if start is None or len(path) < 2:
            return base_ticks
        first_hop = path[1]
        edge_distance = area_distance(self._world_info, start, first_hop)
        distance_per_extra_tick = max(1.0, float(setting('path.recovery.edge_distance_per_extra_tick', 30000.0)))
        extra_ticks = int(edge_distance / distance_per_extra_tick)
        return max(base_ticks, min(max_ticks, base_ticks + extra_ticks))

    def _get_cached(
        self,
        start: Optional[EntityID],
        goal: Optional[EntityID],
        request: dict[str, object],
        runtime_state,
        tick: int,
        avoid_blocked_first_hops: bool,
    ) -> Optional[list[EntityID]]:
        if not bool(setting('path.cache.enabled', True)):
            return None
        key = self._cache_key(start, goal)
        if key not in self._cache:
            return None
        cached_tick, cached_path, cached_result = self._cache[key]
        ttl = int(setting('path.cache.ttl_ticks', 2))
        if tick - cached_tick > ttl:
            self._cache.pop(key, None)
            return None
        first_hop = cached_path[1] if len(cached_path) >= 2 else None
        if avoid_blocked_first_hops and first_hop is not None and runtime_state.is_first_hop_blocked(start, first_hop, tick):
            self._decision_logger.debug_event(
                'cache_skip_blocked_first_hop',
                {
                    'start': entity_value(start),
                    'goal': entity_value(goal),
                    'first_hop': entity_value(first_hop),
                    'caller_context': request.get('caller_context'),
                },
            )
            return None
        self._cache.move_to_end(key)
        result = dict(cached_result)
        result['status'] = 'cache_hit'
        result['cache_hit'] = True
        self._decision_logger.path_snapshot(request, result)
        self._decision_logger.log_text(
            'Путь взят из кэша',
            {
                'цель': key[1],
                'длина_пути': len(cached_path),
                'дистанция': result.get('distance'),
                'caller': request.get('caller_context'),
            },
        )
        runtime_state.last_path_goal = goal
        runtime_state.last_path_distance = float(result.get('distance', 0.0) or 0.0)
        runtime_state.last_path_tick = tick
        return list(cached_path)

    def _put_cache(self, start: Optional[EntityID], goal: Optional[EntityID], path: list[EntityID], result: dict[str, object]) -> None:
        if not bool(setting('path.cache.enabled', True)):
            return
        key = self._cache_key(start, goal)
        self._cache[key] = (self._tick(), list(path), dict(result))
        self._cache.move_to_end(key)
        max_size = int(setting('path.cache.max_size', 512))
        while len(self._cache) > max_size:
            self._cache.popitem(last=False)

    def _step_penalty(self, node_id: EntityID) -> float:
        node = self._world_info.get_entity(node_id)
        if isinstance(node, Road):
            return -float(setting('path.weights.road_preference_bonus', 120.0))
        if isinstance(node, Building):
            return float(setting('path.weights.building_penalty', 80.0))
        return 0.0


    def _path_ml_enabled(self) -> bool:
        return self._ml.mode_name() in {'shadow', 'hybrid', 'pure_ml_test'} and bool(setting('path.ml.apply_only_on_first_hop', True))

    def _path_ml_shadow_only(self) -> bool:
        return self._ml.mode_name() == 'shadow' or bool(setting('path.ml.shadow_only', False))

    def _path_ml_context_allowed(self, caller_context: str) -> bool:
        if caller_context == 'search':
            return bool(setting('path.ml.allow_search_context', True))
        if caller_context == 'detector':
            return bool(setting('path.ml.allow_detector_context', True))
        return bool(setting('path.ml.allow_action_context', True))

    def _build_path_ml_context(self, runtime_state, caller_context: str, startup_recovery_locked: bool, goal: Optional[EntityID] = None) -> dict[str, object]:
        return {
            'phase': runtime_state.phase,
            'caller_context': caller_context,
            'stationary_ticks': runtime_state.stationary_ticks,
            'startup_recovery_locked': startup_recovery_locked,
            'same_goal_as_last_path': 1.0 if (goal is not None and runtime_state.last_path_goal == goal) else 0.0,
        }

    def _build_path_ml_payload(
        self,
        start: EntityID,
        goal: EntityID,
        first_hop: Optional[EntityID],
        path: list[EntityID],
        meta: dict[str, object],
        runtime_state,
        tick: int,
        caller_context: str,
        cache_hit: bool = False,
    ) -> dict[str, object]:
        path_info = summarize_path(self._world_info, path)
        blocked_active_count = runtime_state.count_blocked_first_hops_from(start, tick)
        failed_first_hop_count = 0
        if first_hop is not None:
            failed_first_hop_count = int(runtime_state.failed_first_hops.get(self._edge_key(start, first_hop), 0))
        payload: dict[str, object] = {
            'start': entity_value(start),
            'goal': entity_value(goal),
            'first_hop': entity_value(first_hop),
            'path_distance': float(path_info.get('distance', 0.0) or 0.0),
            'node_count': int(path_info.get('node_count', 0) or 0),
            'road_nodes': int(path_info.get('road_nodes', 0) or 0),
            'building_nodes': int(path_info.get('building_nodes', 0) or 0),
            'expanded': int(meta.get('expanded', 0) or 0),
            'stationary_ticks': int(runtime_state.stationary_ticks),
            'startup_recovery_locked': 1.0 if bool(meta.get('startup_recovery_locked', False)) else 0.0,
            'blocked_first_hops_active_count': blocked_active_count,
            'skipped_start_edges_count': len(meta.get('skipped_start_edges', []) or []),
            'failed_first_hop_count': failed_first_hop_count,
            'first_hop_preblocked': 1.0 if first_hop is not None and runtime_state.is_first_hop_blocked(start, first_hop, tick) else 0.0,
            'first_hop_backtracks': 1.0 if first_hop is not None and runtime_state.previous_position == first_hop else 0.0,
            'same_goal_as_last_path': 1.0 if runtime_state.last_path_goal == goal else 0.0,
            'caller_context': caller_context,
            'phase': runtime_state.phase,
            'cache_hit': 1.0 if cache_hit else 0.0,
        }
        return payload

    def _candidate_final_cost(self, base_cost: float, risk: float | None, runtime_state, start: EntityID, first_hop: Optional[EntityID]) -> float:
        total = float(base_cost)
        if risk is not None:
            risk_weight = float(setting('path.ml.risk_weight', 12000.0))
            risk_value = max(0.0, float(risk))
            high_risk_threshold = float(setting('path.ml.high_risk_threshold', 0.55))
            high_risk_multiplier = max(1.0, float(setting('path.ml.high_risk_penalty_multiplier', 1.6)))
            if risk_value >= high_risk_threshold:
                total += risk_weight * risk_value * high_risk_multiplier
            else:
                total += risk_weight * risk_value
        if first_hop is not None:
            fail_count = int(runtime_state.failed_first_hops.get(self._edge_key(start, first_hop), 0))
            total += float(setting('path.ml.history_fail_penalty', 4000.0)) * float(fail_count)
            if runtime_state.previous_position == first_hop:
                total += float(setting('path.ml.backtrack_penalty', 2500.0))
            if runtime_state.is_first_hop_blocked(start, first_hop, self._tick()):
                total += float(setting('path.ml.blocked_history_penalty', 5000.0))
        return total

    def _path_override_context_allowed(self, caller_context: str) -> bool:
        if caller_context == 'search':
            return bool(setting('path.ml.allow_search_override', True))
        if caller_context == 'detector':
            return bool(setting('path.ml.allow_detector_override', True))
        return bool(setting('path.ml.allow_action_override', True))

    def _safe_risk_value(self, candidate: dict[str, object]) -> float:
        value = candidate.get('risk')
        if value is None:
            return 1.0
        try:
            return max(0.0, float(value))
        except Exception:
            return 1.0

    def _select_ml_override_candidate(
        self,
        baseline_candidate: dict[str, object],
        challengers: list[dict[str, object]],
        caller_context: str,
    ) -> tuple[Optional[dict[str, object]], Optional[dict[str, object]], str, dict[str, object]]:
        debug: dict[str, object] = {'logic_version': PATH_LOGIC_VERSION, 'selection_mode': 'none'}
        if not challengers:
            debug['would_override_if_enabled'] = False
            debug['would_override_reason_if_enabled'] = 'no_challengers'
            debug['override_runtime_blocked_reason'] = 'no_challengers'
            return None, None, 'no_challengers', debug
        best_by_risk = min(challengers, key=lambda item: (self._safe_risk_value(item), float(item.get('final_cost', float('inf')))))
        best_by_final = min(challengers, key=lambda item: float(item.get('final_cost', float('inf'))))
        debug['best_by_risk_first_hop'] = entity_value(best_by_risk.get('first_hop'))
        debug['best_by_final_first_hop'] = entity_value(best_by_final.get('first_hop'))

        allowed, reason, extra = self._should_override_first_hop(baseline_candidate, best_by_risk, caller_context)
        merged = dict(debug)
        merged.update(extra)
        merged['selection_mode'] = 'best_by_risk'
        if allowed:
            return best_by_risk, best_by_final, reason, merged

        allowed_final, reason_final, extra_final = self._should_override_first_hop(baseline_candidate, best_by_final, caller_context)
        merged_final = dict(debug)
        merged_final.update(extra_final)
        merged_final['selection_mode'] = 'best_by_final'
        if allowed_final:
            return best_by_final, best_by_final, reason_final, merged_final

        if bool(merged.get('would_override_if_enabled', False)):
            return None, best_by_final, reason, merged
        if bool(merged_final.get('would_override_if_enabled', False)):
            return None, best_by_final, reason_final, merged_final
        return None, best_by_final, reason, merged

    def _candidate_summary(self, candidate: dict[str, object]) -> dict[str, object]:
        return {
            'first_hop': entity_value(candidate.get('first_hop')),
            'risk': None if candidate.get('risk') is None else round(float(candidate['risk']), 6),
            'base_cost': round(float(candidate.get('base_cost', 0.0) or 0.0), 3),
            'final_cost': round(float(candidate.get('final_cost', 0.0) or 0.0), 3),
            'distance': round(float(candidate.get('distance', 0.0) or 0.0), 3),
            'backtracks': bool(candidate.get('backtracks', False)),
            'blocked_history': bool(candidate.get('blocked_history', False)),
            'fail_count': int(candidate.get('fail_count', 0) or 0),
        }

    def _should_override_first_hop(
        self,
        baseline_candidate: dict[str, object],
        challenger_candidate: dict[str, object],
        caller_context: str,
    ) -> tuple[bool, str, dict[str, object]]:
        debug: dict[str, object] = {'logic_version': PATH_LOGIC_VERSION}
        config_enabled = bool(setting('path.ml.enable_first_hop_override', True))
        context_allowed = self._path_override_context_allowed(caller_context)
        debug['override_enabled_config'] = config_enabled
        debug['override_context_allowed'] = context_allowed

        baseline_first_hop = entity_value(baseline_candidate.get('first_hop'))
        challenger_first_hop = entity_value(challenger_candidate.get('first_hop'))
        debug['baseline_first_hop'] = baseline_first_hop
        debug['challenger_first_hop'] = challenger_first_hop
        if baseline_first_hop == challenger_first_hop:
            debug['would_override_if_enabled'] = False
            debug['would_override_reason_if_enabled'] = 'same_first_hop'
            debug['override_runtime_blocked_reason'] = None
            return False, 'same_first_hop', debug

        baseline_risk_value = self._safe_risk_value(baseline_candidate)
        challenger_risk_value = self._safe_risk_value(challenger_candidate)
        risk_improvement = baseline_risk_value - challenger_risk_value
        risk_ratio = (challenger_risk_value / baseline_risk_value) if baseline_risk_value > 1e-9 else 1.0
        debug['baseline_risk'] = round(baseline_risk_value, 6)
        debug['challenger_risk'] = round(challenger_risk_value, 6)
        debug['risk_improvement'] = round(risk_improvement, 6)
        debug['risk_ratio'] = round(risk_ratio, 6)

        baseline_gate = float(setting('path.ml.override_baseline_risk_threshold', 0.045))
        relative_gate = float(setting('path.ml.override_relative_baseline_risk_threshold', 0.02))
        min_abs_improvement = float(setting('path.ml.override_min_risk_improvement', 0.008))
        min_rel_ratio = float(setting('path.ml.override_max_risk_ratio', 0.78))
        strong_abs_improvement = float(setting('path.ml.override_strong_risk_improvement', 0.018))
        similar_cost_only_improvement = float(setting('path.ml.override_similar_cost_min_risk_improvement', 0.004))

        primary_gate = baseline_risk_value >= baseline_gate and risk_improvement >= min_abs_improvement
        relative_gate_ok = baseline_risk_value >= relative_gate and risk_improvement >= float(setting('path.ml.override_relative_min_risk_improvement', 0.005)) and risk_ratio <= min_rel_ratio
        very_safe_debug_gate = bool(setting('path.ml.debug_force_override_when_very_safe', False)) and baseline_risk_value >= float(setting('path.ml.debug_force_min_baseline_risk', 0.20)) and risk_improvement >= float(setting('path.ml.debug_force_min_risk_improvement', 0.05)) and risk_ratio <= float(setting('path.ml.debug_force_max_risk_ratio', 0.70))
        debug['primary_gate'] = primary_gate
        debug['relative_gate'] = relative_gate_ok
        debug['very_safe_debug_gate'] = very_safe_debug_gate
        debug['baseline_risk_high'] = baseline_risk_value >= baseline_gate
        debug['relative_baseline_risk_high'] = baseline_risk_value >= relative_gate

        challenger_base_cost = float(challenger_candidate.get('base_cost', 0.0) or 0.0)
        baseline_base_cost = max(1.0, float(baseline_candidate.get('base_cost', 0.0) or 0.0))
        cost_ratio = challenger_base_cost / baseline_base_cost
        cost_delta = challenger_base_cost - baseline_base_cost
        debug['base_cost_ratio'] = round(cost_ratio, 6)
        debug['base_cost_delta'] = round(cost_delta, 3)
        debug['cost_ratio_vs_baseline'] = round(cost_ratio, 6)
        debug['cost_delta_vs_baseline'] = round(cost_delta, 3)

        blocked_history_detected = bool(challenger_candidate.get('blocked_history', False))
        backtrack_detected = bool(challenger_candidate.get('backtracks', False))
        higher_fail_count_detected = int(challenger_candidate.get('fail_count', 0) or 0) > int(baseline_candidate.get('fail_count', 0) or 0)
        debug['blocked_history_detected'] = blocked_history_detected
        debug['backtrack_detected'] = backtrack_detected
        debug['higher_fail_count_detected'] = higher_fail_count_detected

        baseline_final_cost = float(baseline_candidate.get('final_cost', 0.0) or 0.0)
        challenger_final_cost = float(challenger_candidate.get('final_cost', 0.0) or 0.0)
        final_ratio = challenger_final_cost / max(1.0, baseline_final_cost)
        debug['baseline_final_cost'] = round(baseline_final_cost, 3)
        debug['challenger_final_cost'] = round(challenger_final_cost, 3)
        debug['final_cost_ratio'] = round(final_ratio, 6)
        debug['final_cost_ratio_vs_baseline'] = round(final_ratio, 6)
        max_final_cost_ratio = max(1.0, float(setting('path.ml.override_max_final_cost_ratio', 1.16)))
        relaxed_ratio = max(1.0, float(setting('path.ml.override_relaxed_final_cost_ratio', 1.09)))
        debug_force_ratio = max(1.0, float(setting('path.ml.debug_force_max_final_cost_ratio', 1.02)))
        strong_risk_override = risk_improvement >= strong_abs_improvement
        debug['strong_risk_override'] = strong_risk_override
        near_cost_tie = final_ratio <= max(1.0, float(setting('path.ml.override_near_tie_final_cost_ratio', 1.02)))
        debug['near_cost_tie'] = near_cost_tie

        gate_pass = False
        reason_if_enabled = 'risk_gate_not_met'
        runtime_blocked_reason = None

        if challenger_final_cost <= baseline_final_cost and risk_improvement > 0.0:
            gate_pass = True
            reason_if_enabled = 'lower_final_cost'
        elif near_cost_tie and risk_improvement >= similar_cost_only_improvement:
            gate_pass = True
            reason_if_enabled = 'near_tie_lower_risk'
        elif not (primary_gate or relative_gate_ok or very_safe_debug_gate):
            gate_pass = False
            reason_if_enabled = 'risk_gate_not_met'
        elif challenger_base_cost > baseline_base_cost * max(1.0, float(setting('path.ml.override_max_cost_ratio', 1.28))):
            gate_pass = False
            reason_if_enabled = 'cost_ratio_too_high'
        elif cost_delta > float(setting('path.ml.override_max_cost_delta', 32000.0)):
            gate_pass = False
            reason_if_enabled = 'cost_delta_too_high'
        elif blocked_history_detected:
            gate_pass = False
            reason_if_enabled = 'challenger_blocked_history'
        elif backtrack_detected and not bool(setting('path.ml.override_allow_backtrack', False)):
            gate_pass = False
            reason_if_enabled = 'challenger_backtracks'
        elif higher_fail_count_detected and not bool(setting('path.ml.override_allow_higher_fail_count', False)):
            gate_pass = False
            reason_if_enabled = 'challenger_higher_fail_count'
        elif challenger_final_cost > max(1.0, baseline_final_cost) * max_final_cost_ratio and not very_safe_debug_gate:
            gate_pass = False
            reason_if_enabled = 'final_cost_ratio_too_high'
        elif strong_risk_override and challenger_final_cost <= max(1.0, baseline_final_cost) * relaxed_ratio:
            gate_pass = True
            reason_if_enabled = 'strong_risk_improvement'
        elif relative_gate_ok and challenger_final_cost <= max(1.0, baseline_final_cost) * relaxed_ratio:
            gate_pass = True
            reason_if_enabled = 'relative_risk_override'
        elif very_safe_debug_gate and challenger_final_cost <= max(1.0, baseline_final_cost) * debug_force_ratio:
            gate_pass = True
            reason_if_enabled = 'debug_safe_override'
        elif challenger_final_cost <= max(1.0, baseline_final_cost) * relaxed_ratio and risk_improvement > 0.0:
            gate_pass = True
            reason_if_enabled = 'safe_risk_override'
        else:
            gate_pass = False
            reason_if_enabled = 'final_cost_still_worse'

        if gate_pass:
            if not config_enabled:
                runtime_blocked_reason = 'override_disabled'
            elif not context_allowed:
                runtime_blocked_reason = 'override_context_disabled'

        debug['safe_gate_passed'] = gate_pass
        debug['would_override_if_enabled'] = gate_pass
        debug['would_override_reason_if_enabled'] = reason_if_enabled
        debug['override_runtime_blocked_reason'] = runtime_blocked_reason

        actual_allowed = gate_pass and runtime_blocked_reason is None
        actual_reason = reason_if_enabled if actual_allowed else (runtime_blocked_reason or reason_if_enabled)
        return actual_allowed, actual_reason, debug

    def _build_forced_candidate(
        self,
        start: EntityID,
        goal: EntityID,
        first_hop: EntityID,
        runtime_state,
        tick: int,
        caller_context: str,
        startup_recovery_locked: bool,
    ) -> Optional[dict[str, object]]:
        if first_hop == goal:
            tail_path = [goal]
            tail_meta: dict[str, object] = {'expanded': 0, 'skipped_start_edges': [], 'total_cost': 0.0}
        else:
            tail_path, tail_meta = self._run_a_star(first_hop, goal, runtime_state, tick, False)
            if not tail_path:
                return None
        path = [start] + list(tail_path)
        base_cost = float(area_distance(self._world_info, start, first_hop) + self._step_penalty(first_hop) + float(tail_meta.get('total_cost', 0.0) or 0.0))
        payload = self._build_path_ml_payload(
            start,
            goal,
            first_hop,
            path,
            {**tail_meta, 'startup_recovery_locked': startup_recovery_locked},
            runtime_state,
            tick,
            caller_context,
            cache_hit=False,
        )
        context = self._build_path_ml_context(runtime_state, caller_context, startup_recovery_locked, goal)
        risk = self._ml.score_path(context, payload)
        final_cost = self._candidate_final_cost(base_cost, risk, runtime_state, start, first_hop)
        return {
            'path': path,
            'meta': tail_meta,
            'payload': payload,
            'risk': risk,
            'base_cost': base_cost,
            'final_cost': final_cost,
            'first_hop': first_hop,
            'distance': float(payload.get('path_distance', 0.0) or 0.0),
            'backtracks': bool(runtime_state.previous_position == first_hop),
            'blocked_history': bool(runtime_state.is_first_hop_blocked(start, first_hop, tick)),
            'fail_count': int(runtime_state.failed_first_hops.get(self._edge_key(start, first_hop), 0)),
        }

    def _rerank_first_hop(
        self,
        start: EntityID,
        goal: EntityID,
        baseline_path: list[EntityID],
        baseline_meta: dict[str, object],
        runtime_state,
        tick: int,
        caller_context: str,
        startup_recovery_locked: bool,
        avoid_blocked_first_hops: bool,
    ) -> tuple[list[EntityID], dict[str, object], dict[str, object]]:
        diagnostics: dict[str, object] = {
            'ml_rerank_active': False,
            'path_logic_version': PATH_LOGIC_VERSION,
            'shadow_only': self._path_ml_shadow_only(),
            'override_gate_reached': False,
            'override_candidates_built': 0,
            'override_evaluation_done': False,
            'would_override': False,
            'would_override_reason': None,
        }
        if not self._path_ml_enabled() or not self._path_ml_context_allowed(caller_context):
            return baseline_path, baseline_meta, diagnostics
        if len(baseline_path) < 2:
            return baseline_path, baseline_meta, diagnostics
        baseline_first_hop = baseline_path[1]
        baseline_payload = self._build_path_ml_payload(
            start,
            goal,
            baseline_first_hop,
            baseline_path,
            {**baseline_meta, 'startup_recovery_locked': startup_recovery_locked},
            runtime_state,
            tick,
            caller_context,
            cache_hit=False,
        )
        context = self._build_path_ml_context(runtime_state, caller_context, startup_recovery_locked, goal)
        baseline_risk = self._ml.score_path(context, baseline_payload)
        baseline_base_cost = float(baseline_meta.get('total_cost', path_distance(self._world_info, baseline_path)) or 0.0)
        baseline_final_cost = self._candidate_final_cost(baseline_base_cost, baseline_risk, runtime_state, start, baseline_first_hop)
        baseline_candidate: dict[str, object] = {
            'path': baseline_path,
            'meta': baseline_meta,
            'payload': baseline_payload,
            'risk': baseline_risk,
            'base_cost': baseline_base_cost,
            'final_cost': baseline_final_cost,
            'first_hop': baseline_first_hop,
            'distance': float(baseline_payload.get('path_distance', 0.0) or 0.0),
            'backtracks': bool(runtime_state.previous_position == baseline_first_hop),
            'blocked_history': bool(runtime_state.is_first_hop_blocked(start, baseline_first_hop, tick)),
            'fail_count': int(runtime_state.failed_first_hops.get(self._edge_key(start, baseline_first_hop), 0)),
        }
        best_candidate: dict[str, object] = dict(baseline_candidate)
        diagnostics.update({
            'ml_rerank_active': True,
            'baseline_first_hop': entity_value(baseline_first_hop),
            'baseline_risk': None if baseline_risk is None else round(float(baseline_risk), 6),
            'baseline_base_cost': round(float(baseline_base_cost), 3),
            'baseline_final_cost': round(float(baseline_final_cost), 3),
        })
        neighbors = list(self._graph.get(start, []))
        diagnostics['override_gate_reached'] = True
        if not neighbors:
            diagnostics['override_blocked_reason'] = 'no_neighbors'
            return baseline_path, baseline_meta, diagnostics
        max_candidates = max(2, int(setting('path.ml.max_first_hop_candidates', 8)))
        max_distance_ratio = max(1.0, float(setting('path.ml.max_distance_ratio', 1.35)))
        baseline_distance = float(baseline_payload.get('path_distance', 0.0) or 0.0)
        candidate_count = 1
        alt_count = 0
        candidates: list[dict[str, object]] = [baseline_candidate]
        for neighbor in sorted(neighbors, key=lambda node: area_distance(self._world_info, node, goal)):
            if neighbor == baseline_first_hop:
                continue
            if avoid_blocked_first_hops and runtime_state.is_first_hop_blocked(start, neighbor, tick):
                continue
            if alt_count >= max_candidates - 1:
                break
            candidate = self._build_forced_candidate(start, goal, neighbor, runtime_state, tick, caller_context, startup_recovery_locked)
            if candidate is None:
                continue
            alt_count += 1
            candidate_distance = float(candidate['payload'].get('path_distance', 0.0) or 0.0)
            if baseline_distance > 0.0 and candidate_distance > baseline_distance * max_distance_ratio:
                continue
            candidate_count += 1
            candidates.append(candidate)
            if float(candidate['final_cost']) < float(best_candidate['final_cost']):
                best_candidate = candidate

        candidates_sorted = sorted(candidates, key=lambda item: float(item.get('final_cost', float('inf'))))
        challengers = [candidate for candidate in candidates if entity_value(candidate.get('first_hop')) != entity_value(baseline_first_hop)]
        diagnostics['override_candidates_built'] = len(challengers)
        override_candidate, ml_best_candidate, best_attempted_reason, override_debug = self._select_ml_override_candidate(
            baseline_candidate,
            challengers,
            caller_context,
        )
        if ml_best_candidate is None:
            ml_best_candidate = dict(candidates_sorted[0]) if candidates_sorted else dict(baseline_candidate)
        diagnostics['override_evaluation_done'] = True
        diagnostics['candidate_count'] = candidate_count
        diagnostics['ml_best_first_hop'] = entity_value(ml_best_candidate.get('first_hop'))
        diagnostics['ml_best_risk'] = None if ml_best_candidate.get('risk') is None else round(float(ml_best_candidate['risk']), 6)
        diagnostics['ml_best_base_cost'] = round(float(ml_best_candidate.get('base_cost', 0.0) or 0.0), 3)
        diagnostics['ml_best_final_cost'] = round(float(ml_best_candidate.get('final_cost', 0.0) or 0.0), 3)
        diagnostics['would_override_if_enabled'] = bool(override_debug.get('would_override_if_enabled', False))
        diagnostics['would_override'] = diagnostics['would_override_if_enabled']
        diagnostics['would_override_reason_if_enabled'] = override_debug.get('would_override_reason_if_enabled')
        diagnostics['would_override_reason'] = override_debug.get('would_override_reason_if_enabled')
        diagnostics['override_runtime_blocked_reason'] = override_debug.get('override_runtime_blocked_reason')

        selected_candidate = baseline_candidate
        override_reason = None
        if not self._path_ml_shadow_only() and override_candidate is not None:
            selected_candidate = override_candidate
            override_reason = best_attempted_reason

        diagnostics['chosen_first_hop'] = entity_value(selected_candidate.get('first_hop'))
        diagnostics['chosen_risk'] = None if selected_candidate.get('risk') is None else round(float(selected_candidate['risk']), 6)
        diagnostics['chosen_base_cost'] = round(float(selected_candidate.get('base_cost', 0.0) or 0.0), 3)
        diagnostics['chosen_final_cost'] = round(float(selected_candidate.get('final_cost', 0.0) or 0.0), 3)
        diagnostics['override_applied'] = override_candidate is not None and not self._path_ml_shadow_only()
        diagnostics['override_reason'] = override_reason
        diagnostics['override_blocked_reason'] = override_debug.get('override_runtime_blocked_reason') if not diagnostics['override_applied'] else None
        diagnostics['override_debug'] = override_debug
        diagnostics['risk_improvement'] = override_debug.get('risk_improvement')
        diagnostics['baseline_risk_high'] = override_debug.get('baseline_risk_high')
        diagnostics['cost_ratio_vs_baseline'] = override_debug.get('cost_ratio_vs_baseline')
        diagnostics['backtrack_detected'] = override_debug.get('backtrack_detected')
        diagnostics['blocked_history_detected'] = override_debug.get('blocked_history_detected')
        diagnostics['risk_delta_vs_baseline'] = None if selected_candidate.get('risk') is None or baseline_risk is None else round(float(baseline_risk) - float(selected_candidate.get('risk') or 0.0), 6)
        diagnostics['first_hop_changed_by_ml'] = entity_value(selected_candidate.get('first_hop')) != entity_value(baseline_first_hop)
        shadow_top_n = max(2, min(int(setting('path.ml.shadow_log_top_candidates', 6)), len(candidates_sorted)))
        diagnostics['candidate_summaries'] = [self._candidate_summary(candidate) for candidate in candidates_sorted[:shadow_top_n]]
        diagnostics['baseline_summary'] = self._candidate_summary(baseline_candidate)
        diagnostics['selected_summary'] = self._candidate_summary(selected_candidate)
        diagnostics['ml_best_summary'] = self._candidate_summary(ml_best_candidate)
        return list(selected_candidate['path']), dict(selected_candidate['meta']), diagnostics

    def _run_a_star(self, start: EntityID, goal: EntityID, runtime_state, tick: int, avoid_blocked_first_hops: bool) -> tuple[list[EntityID], dict[str, object]]:
        open_heap: list[tuple[float, int, EntityID]] = []
        serial = count()
        heapq.heappush(open_heap, (0.0, next(serial), start))
        came_from: dict[EntityID, EntityID] = {}
        g_score: dict[EntityID, float] = {start: 0.0}
        expanded = 0
        closed: set[EntityID] = set()
        skipped_start_edges: list[str] = []
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expanded += 1
            if current == goal:
                return reconstruct_path(came_from, goal), {'expanded': expanded, 'skipped_start_edges': skipped_start_edges, 'total_cost': g_score.get(goal, 0.0)}
            for neighbor in self._graph.get(current, []):
                if current == start and avoid_blocked_first_hops and runtime_state.is_first_hop_blocked(start, neighbor, tick):
                    skipped_start_edges.append(self._edge_key(start, neighbor))
                    continue
                tentative = g_score[current] + area_distance(self._world_info, current, neighbor) + self._step_penalty(neighbor)
                if tentative < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    heuristic = area_distance(self._world_info, neighbor, goal)
                    heapq.heappush(open_heap, (tentative + heuristic, next(serial), neighbor))
        return [], {'expanded': expanded, 'skipped_start_edges': skipped_start_edges, 'total_cost': None}

    def get_path(self, from_entity_id: EntityID, to_entity_id: EntityID) -> list[EntityID]:
        started = time.monotonic()
        start = self._normalize(from_entity_id)
        goal = self._normalize(to_entity_id)
        tick = self._tick()
        runtime_state = get_runtime_state(self._agent_info)
        caller_context = self._caller_context()
        action_context = self._is_action_context(caller_context)
        startup_recovery_locked = bool(setting('path.recovery.activate_only_after_first_real_move', True)) and not runtime_state.has_real_movement()
        ml_info = self._ml.describe()
        request = {
            'from': entity_value(start),
            'to': entity_value(goal),
            'mode': self._ml.mode_name(),
            'phase': runtime_state.phase,
            'caller_context': caller_context,
            'stationary_ticks': runtime_state.stationary_ticks,
            'startup_recovery_locked': startup_recovery_locked,
            'first_real_move_tick': runtime_state.first_real_move_tick,
            'path_ml_active': self._path_ml_enabled(),
            'path_ml_shadow_only': self._path_ml_shadow_only(),
            'path_logic_version': PATH_LOGIC_VERSION,
            'configured_model_path': ml_info.get('configured_model_path'),
            'resolved_model_path': ml_info.get('resolved_model_path'),
            'model_exists': ml_info.get('model_exists'),
            'model_size_bytes': ml_info.get('model_size_bytes'),
            'model_mtime': ml_info.get('model_mtime'),
        }

        if action_context:
            if startup_recovery_locked:
                self._decision_logger.debug_event(
                    'startup_recovery_grace_active',
                    {
                        'tick': tick,
                        'start': entity_value(start),
                        'goal': entity_value(goal),
                        'caller_context': caller_context,
                        'startup_position': entity_value(runtime_state.startup_position),
                        'first_real_move_tick': runtime_state.first_real_move_tick,
                        'stationary_ticks': runtime_state.stationary_ticks,
                    },
                )
            else:
                outcome = runtime_state.resolve_pending_move_outcome(
                    self._agent_info.get_position_entity_id(),
                    tick,
                    failure_threshold=max(1, int(setting('path.recovery.first_hop_failure_threshold', 2))),
                    block_ticks=max(1, int(setting('path.recovery.first_hop_block_ticks', 8))),
                )
                if outcome is not None:
                    self._decision_logger.debug_event('move_outcome_resolved', outcome)
                    if outcome.get('outcome') == 'stalled':
                        self._decision_logger.log_text(
                            'Обнаружен провал первого шага пути',
                            {
                                'edge': outcome.get('edge_key'),
                                'fail_count': outcome.get('fail_count'),
                                'blocked_until': outcome.get('blocked_until'),
                                'goal': outcome.get('goal'),
                                'caller': caller_context,
                            },
                        )
                    else:
                        self._decision_logger.debug_text('Движение по прошлой попытке подтвердилось', outcome)

        if start is None or goal is None:
            result = {'status': 'invalid', 'cache_hit': False, 'path': [], 'distance': None, 'expanded': 0, 'compute_ms': 0.0, 'caller_context': caller_context}
            self._decision_logger.log_text('Построение пути отменено: не удалось определить старт или цель', request)
            self._decision_logger.path_snapshot(request, result)
            return []

        avoid_blocked_first_hops = bool(setting('path.recovery.avoid_blocked_first_hops', True)) and not startup_recovery_locked
        cached = self._get_cached(start, goal, request, runtime_state, tick, avoid_blocked_first_hops)
        if cached is not None:
            if action_context and len(cached) >= 2:
                runtime_state.note_path_attempt(start, goal, cached, tick, caller_context, grace_ticks=self._estimate_move_grace_ticks(start, cached))
                self._decision_logger.debug_event(
                    'move_attempt_registered',
                    {
                        'source': 'cache',
                        'start': entity_value(start),
                        'goal': entity_value(goal),
                        'first_hop': entity_value(cached[1] if len(cached) >= 2 else None),
                        'path_nodes': len(cached),
                        'caller_context': caller_context,
                        'startup_recovery_locked': startup_recovery_locked,
                        'grace_ticks': self._estimate_move_grace_ticks(start, cached),
                    },
                )
            return cached

        stuck_threshold = int(setting('path.stuck.stationary_ticks', 6))
        if runtime_state.stationary_ticks >= stuck_threshold and runtime_state.last_path_goal == goal and runtime_state.last_position == start:
            self._decision_logger.debug_event(
                'soft_stuck_notice',
                {
                    'start': entity_value(start),
                    'goal': entity_value(goal),
                    'stationary_ticks': runtime_state.stationary_ticks,
                    'caller_context': caller_context,
                },
            )

        if start == goal:
            result = {
                'status': 'trivial',
                'cache_hit': False,
                'path': [entity_value(start)],
                'distance': 0.0,
                'expanded': 0,
                'compute_ms': 0.0,
                'node_count': 1,
                'road_nodes': 0,
                'building_nodes': 1,
                'first_hop': None,
                'caller_context': caller_context,
            }
            self._decision_logger.log_text('Путь не требуется: агент уже в целевой зоне', request)
            self._decision_logger.path_snapshot(request, result)
            self._put_cache(start, goal, [start], result)
            runtime_state.last_path_goal = goal
            runtime_state.last_path_distance = 0.0
            runtime_state.last_path_tick = tick
            return [start]

        try:
            path, meta = self._run_a_star(start, goal, runtime_state, tick, avoid_blocked_first_hops)
            if path:
                path, meta, ml_first_hop_info = self._rerank_first_hop(
                    start,
                    goal,
                    path,
                    meta,
                    runtime_state,
                    tick,
                    caller_context,
                    startup_recovery_locked,
                    avoid_blocked_first_hops,
                )
            else:
                ml_first_hop_info = {'ml_rerank_active': False}
            compute_ms = (time.monotonic() - started) * 1000.0
            if path:
                first_hop = path[1] if len(path) >= 2 else None
                if first_hop is not None and runtime_state.is_first_hop_blocked(start, first_hop, tick):
                    self._decision_logger.debug_event(
                        'reject_path_blocked_first_hop',
                        {
                            'start': entity_value(start),
                            'goal': entity_value(goal),
                            'first_hop': entity_value(first_hop),
                            'caller_context': caller_context,
                            'blocked_first_hops_active': runtime_state.blocked_first_hops_from(start, tick),
                        },
                    )
                    path = []
                else:
                    path_info = summarize_path(self._world_info, path)
                    result = {
                        'status': 'ok',
                        'cache_hit': False,
                        'path': [entity_value(node) for node in path],
                        'distance': path_info['distance'],
                        'expanded': meta.get('expanded', 0),
                        'compute_ms': round(compute_ms, 3),
                        'node_count': path_info['node_count'],
                        'road_nodes': path_info['road_nodes'],
                        'building_nodes': path_info['building_nodes'],
                        'first_hop': entity_value(first_hop),
                        'caller_context': caller_context,
                        'skipped_start_edges': list(meta.get('skipped_start_edges', [])),
                        'blocked_first_hops_active': runtime_state.blocked_first_hops_from(start, tick),
                        'ml_first_hop_info': ml_first_hop_info,
                    }
                    self._decision_logger.path_snapshot(request, result)
                    self._decision_logger.log_text(
                        'Путь построен',
                        {
                            'цель': entity_value(goal),
                            'длина_пути': path_info['node_count'],
                            'дистанция': path_info['distance'],
                            'дорог': path_info['road_nodes'],
                            'зданий': path_info['building_nodes'],
                            'раскрыто_узлов': meta.get('expanded', 0),
                            'время_мс': round(compute_ms, 3),
                            'first_hop': entity_value(first_hop),
                            'caller': caller_context,
                        },
                    )
                    self._put_cache(start, goal, path, result)
                    runtime_state.last_path_goal = goal
                    runtime_state.last_path_distance = float(path_info['distance'])
                    runtime_state.last_path_tick = tick
                    if action_context and len(path) >= 2:
                        runtime_state.note_path_attempt(start, goal, path, tick, caller_context, grace_ticks=self._estimate_move_grace_ticks(start, path))
                        self._decision_logger.debug_event(
                            'move_attempt_registered',
                            {
                                'source': 'fresh',
                                'start': entity_value(start),
                                'goal': entity_value(goal),
                                'first_hop': entity_value(first_hop),
                                'path_nodes': len(path),
                                'caller_context': caller_context,
                                'blocked_first_hops_active': runtime_state.blocked_first_hops_from(start, tick),
                                'startup_recovery_locked': startup_recovery_locked,
                                'grace_ticks': self._estimate_move_grace_ticks(start, path),
                            },
                        )
                    return path

            result = {
                'status': 'unreachable',
                'cache_hit': False,
                'path': [],
                'distance': None,
                'expanded': meta.get('expanded', 0),
                'compute_ms': round(compute_ms, 3),
                'node_count': 0,
                'road_nodes': 0,
                'building_nodes': 0,
                'first_hop': None,
                'caller_context': caller_context,
                'skipped_start_edges': list(meta.get('skipped_start_edges', [])),
                'blocked_first_hops_active': runtime_state.blocked_first_hops_from(start, tick),
                'ml_first_hop_info': ml_first_hop_info,
            }
            self._decision_logger.path_snapshot(request, result)
            self._decision_logger.log_text(
                'Путь не найден',
                {
                    'цель': entity_value(goal),
                    'раскрыто_узлов': meta.get('expanded', 0),
                    'время_мс': round(compute_ms, 3),
                    'caller': caller_context,
                    'blocked_first_hops_active': runtime_state.count_blocked_first_hops_from(start, tick),
                },
            )
            return []
        except Exception as exc:
            compute_ms = (time.monotonic() - started) * 1000.0
            self._decision_logger.path_snapshot(
                request,
                {
                    'status': 'error',
                    'cache_hit': False,
                    'path': [],
                    'distance': None,
                    'expanded': 0,
                    'compute_ms': round(compute_ms, 3),
                    'error': str(exc),
                    'caller_context': caller_context,
                },
            )
            self._decision_logger.log_text('Ошибка при построении пути', {'ошибка': exc.__class__.__name__, 'сообщение': str(exc), 'caller': caller_context})
            return []

    def get_path_to_multiple_destinations(self, from_entity_id: EntityID, destination_entity_ids: set[EntityID]) -> list[EntityID]:
        best_path: list[EntityID] = []
        best_distance: Optional[float] = None
        for destination in destination_entity_ids:
            path = self.get_path(from_entity_id, destination)
            if not path:
                continue
            distance_value = path_distance(self._world_info, path)
            if best_distance is None or distance_value < best_distance:
                best_distance = distance_value
                best_path = path
        return best_path

    def get_distance(self, from_entity_id: EntityID, to_entity_id: EntityID) -> float:
        return path_distance(self._world_info, self.get_path(from_entity_id, to_entity_id))
