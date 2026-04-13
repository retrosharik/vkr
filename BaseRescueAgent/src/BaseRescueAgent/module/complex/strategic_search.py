from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Optional, cast

from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from adf_core_python.core.component.module.complex.search import Search
from rcrscore.entities import Building, EntityID, Refuge, Road

from ..util.decision_logger import DecisionLogger, entity_value
from ..util.ml_bridge import MlBridge
from ..util.rescue_support import (
    area_distance,
    building_area_value,
    building_centrality,
    civilians_in_building,
    cluster_entities_for_agent,
    get_xy,
    is_search_target_valid,
    refuge_entities,
    searchable_buildings,
    visible_civilians,
)
from ..util.runtime_settings import setting
from ..util.shared_runtime_state import get_runtime_state


class StrategicSearch(Search):
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
                'StrategicSearch.Clustering',
                'adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering',
            ),
        )
        self._path_planning: PathPlanning = cast(
            PathPlanning,
            module_manager.get_module(
                'StrategicSearch.PathPlanning',
                'src.BaseRescueAgent.module.algorithm.logged_a_star_path_planning.LoggedAStarPathPlanning',
            ),
        )
        self.register_sub_module(self._clustering)
        self.register_sub_module(self._path_planning)
        self._visited: set[EntityID] = set()
        self._recent_targets: deque[EntityID] = deque(maxlen=int(setting('search.selection.recent_targets_size', 8)))
        self._recent_progress: deque[bool] = deque(maxlen=int(setting('search.selection.stagnation_window', 6)))
        self._result: Optional[EntityID] = None
        self._decision_logger = DecisionLogger(agent_info, world_info, self.__class__.__name__, 'search')
        self._ml = MlBridge('search')

        self._static_all_buildings: Optional[list[Building]] = None
        self._static_cluster_buildings: Optional[list[Building]] = None
        self._static_cluster_ids: Optional[set[EntityID]] = None
        self._coord_cache: dict[str, tuple[Optional[int], Optional[int]]] = {}
        self._centrality_cache: dict[str, int] = {}
        self._tick_cache_tick: int = -1
        self._distance_cache: dict[tuple[str, str], float] = {}
        self._approx_distance_cache: dict[str, float] = {}
        self._hint_cache_exact: dict[str, int] = {}
        self._hint_cache_fast: dict[str, int] = {}
        self._visible_civilians_cache = []
        self._visible_civilian_positions: dict[str, int] = {}
        self._all_buildings_count = 0
        self._is_large_map = False
        self._agent_xy: tuple[Optional[int], Optional[int]] = (None, None)
        self._last_visited_count = 0
        self._last_area_key: Optional[str] = None
        self._forced_global_until = -1
        self._cluster_revisit_streak = 0
        self._ml_status_logged = False

    def _visited_keys(self, runtime_state) -> set[str]:
        entered = getattr(runtime_state, 'entered_buildings', None)
        if isinstance(entered, set):
            return entered
        visited = getattr(runtime_state, 'visited_buildings', None)
        if isinstance(visited, set):
            return visited
        return set()

    def _is_visited_building(self, runtime_state, building_id: Optional[EntityID]) -> bool:
        key = entity_value(building_id)
        if key == 'None':
            return False
        return key in self._visited_keys(runtime_state)

    def _visited_count(self, runtime_state) -> int:
        return len(self._visited_keys(runtime_state))

    def _ensure_static_map_cache(self) -> None:
        if self._static_all_buildings is None:
            self._static_all_buildings = searchable_buildings(self._world_info)
            self._all_buildings_count = len(self._static_all_buildings)
            self._is_large_map = self._all_buildings_count > int(setting('search.selection.large_map_threshold', 220))
            for building in self._static_all_buildings:
                building_id = building.get_entity_id()
                if building_id is None:
                    continue
                key = entity_value(building_id)
                self._coord_cache[key] = get_xy(building)
        if self._static_cluster_buildings is None:
            me = self._agent_info.get_entity_id()
            cluster_entities = cluster_entities_for_agent(self._clustering, me)
            self._static_cluster_buildings = searchable_buildings(self._world_info, cluster_entities)
            self._static_cluster_ids = {b.get_entity_id() for b in self._static_cluster_buildings if b.get_entity_id() is not None}
            for building in self._static_cluster_buildings:
                building_id = building.get_entity_id()
                if building_id is None:
                    continue
                key = entity_value(building_id)
                if key not in self._coord_cache:
                    self._coord_cache[key] = get_xy(building)

    def _begin_tick_cache(self, tick: int) -> None:
        if self._tick_cache_tick == tick:
            return
        self._tick_cache_tick = tick
        self._distance_cache.clear()
        self._approx_distance_cache.clear()
        self._hint_cache_exact.clear()
        self._hint_cache_fast.clear()
        self._visible_civilians_cache = visible_civilians(self._world_info)
        self._visible_civilian_positions = {}
        for civilian in self._visible_civilians_cache:
            hp = civilian.get_hp() or 0
            if hp <= 0:
                continue
            damage = civilian.get_damage() or 0
            buriedness = civilian.get_buriedness() or 0
            if damage <= 0 and buriedness <= 0:
                continue
            position_id = civilian.get_position()
            if position_id is None:
                continue
            position_entity = self._world_info.get_entity(position_id)
            if not isinstance(position_entity, Building) or isinstance(position_entity, Refuge):
                continue
            key = entity_value(position_id)
            if key == 'None':
                continue
            self._visible_civilian_positions[key] = self._visible_civilian_positions.get(key, 0) + 1
        current_entity = self._world_info.get_entity(self._agent_info.get_position_entity_id())
        self._agent_xy = get_xy(current_entity)

    def _distance_cached(self, from_entity_id: Optional[EntityID], to_entity_id: Optional[EntityID]) -> float:
        key = (entity_value(from_entity_id), entity_value(to_entity_id))
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached
        value = area_distance(self._world_info, from_entity_id, to_entity_id)
        self._distance_cache[key] = value
        return value

    def _approx_distance_to(self, building_id: Optional[EntityID]) -> float:
        key = entity_value(building_id)
        cached = self._approx_distance_cache.get(key)
        if cached is not None:
            return cached
        bx, by = self._coord_cache.get(key, (None, None))
        ax, ay = self._agent_xy
        if ax is None or ay is None or bx is None or by is None:
            value = self._distance_cached(self._agent_info.get_position_entity_id(), building_id)
        else:
            value = math.hypot(float(ax - bx), float(ay - by))
        self._approx_distance_cache[key] = value
        return value

    def _hint_count(self, building_id: Optional[EntityID], fast_only: bool = False) -> int:
        key = entity_value(building_id)
        if key == 'None':
            return 0
        cache = self._hint_cache_fast if fast_only else self._hint_cache_exact
        if key in cache:
            return cache[key]
        if fast_only or self._is_large_map:
            value = int(self._visible_civilian_positions.get(key, 0))
        else:
            value = int(self._visible_civilian_positions.get(key, 0))
            if value <= 0:
                value = 0
                for pos_key, count in self._visible_civilian_positions.items():
                    if count <= 0 or pos_key == 'None':
                        continue
                    try:
                        from rcrscore.entities import EntityID as _EntityID
                        pos_id = _EntityID(int(pos_key))
                    except Exception:
                        continue
                    if self._distance_cached(pos_id, building_id) <= 60000.0:
                        value += count
        cache[key] = value
        return value

    def _centrality(self, building_id: Optional[EntityID]) -> int:
        key = entity_value(building_id)
        if key == 'None':
            return 0
        cached = self._centrality_cache.get(key)
        if cached is not None:
            return cached
        value = int(building_centrality(self._world_info, building_id))
        self._centrality_cache[key] = value
        return value

    def _known_problem_civilians(self, building_id: Optional[EntityID]) -> tuple[int, int]:
        exact = civilians_in_building(self._world_info, building_id, include_stable=True)
        active = 0
        buried = 0
        for civilian in exact:
            hp = civilian.get_hp() or 0
            if hp <= 0:
                continue
            damage = civilian.get_damage() or 0
            buriedness = civilian.get_buriedness() or 0
            if damage > 0 or buriedness > 0:
                active += 1
            if buriedness > 0:
                buried += 1
        return active, buried

    def _required_building_dwell(self, building_id: Optional[EntityID], building: Optional[Building], hint_count: int) -> int:
        area_value = building_area_value(building)
        small_threshold = int(setting('search.inspection.small_building_area_threshold', 30000))
        large_threshold = int(setting('search.inspection.large_building_area_threshold', 70000))
        active_civilians, buried = self._known_problem_civilians(building_id)

        if active_civilians <= 0 and buried <= 0 and hint_count <= 0:
            if area_value >= large_threshold:
                return int(setting('search.inspection.empty_large_building_dwell', 2))
            if area_value >= small_threshold:
                return int(setting('search.inspection.empty_medium_building_dwell', 1))
            return int(setting('search.inspection.empty_small_building_dwell', 1))

        base = int(setting('search.inspection.hinted_base_dwell', 2))
        size_extra = 0
        if area_value >= large_threshold:
            size_extra = int(setting('search.inspection.large_building_extra_dwell', 2))
        elif area_value >= small_threshold:
            size_extra = int(setting('search.inspection.medium_building_extra_dwell', 1))
        hint_extra = min(int(hint_count), int(setting('search.inspection.hint_extra_dwell_cap', 2)))
        active_extra = int(setting('search.inspection.exact_civilian_extra_dwell', 1)) if active_civilians > 0 else 0
        buried_extra = int(setting('search.inspection.buried_extra_dwell', 2)) if buried > 0 else 0
        max_dwell = int(setting('search.inspection.max_dwell_ticks', 8))
        return min(max_dwell, base + size_extra + hint_extra + active_extra + buried_extra)

    def _building_close_info(self, building_id: Optional[EntityID], building: Optional[Building], runtime_state, tick: int, *, dwell_ticks: Optional[int] = None, hint_count: Optional[int] = None) -> dict[str, int | bool | str | None]:
        if building_id is None or building is None:
            return {
                'building': entity_value(building_id),
                'dwell_ticks': 0,
                'required_dwell': 0,
                'hint_count': 0,
                'active_inside': 0,
                'buried_inside': 0,
                'exhausted': False,
            }
        effective_hint_count = self._hint_count(building_id, fast_only=self._is_large_map) if hint_count is None else int(hint_count)
        effective_dwell_ticks = runtime_state.search_building_dwell_ticks(building_id, tick) if dwell_ticks is None else int(dwell_ticks)
        required_dwell = self._required_building_dwell(building_id, building, effective_hint_count)
        active_inside, buried_inside = self._known_problem_civilians(building_id)
        exhausted = effective_dwell_ticks >= required_dwell and active_inside <= 0 and buried_inside <= 0 and effective_hint_count <= 0
        return {
            'building': entity_value(building_id),
            'dwell_ticks': effective_dwell_ticks,
            'required_dwell': required_dwell,
            'hint_count': effective_hint_count,
            'active_inside': active_inside,
            'buried_inside': buried_inside,
            'exhausted': exhausted,
        }

    def _should_close_building(self, building_id: Optional[EntityID], dwell_ticks: int, required_dwell: int, runtime_state, tick: int) -> bool:
        if building_id is None:
            return False
        active_civilians, buried = self._known_problem_civilians(building_id)
        hint_count = self._hint_count(building_id, fast_only=self._is_large_map)
        max_dwell = int(setting('search.inspection.max_dwell_ticks', 8))
        if dwell_ticks < required_dwell:
            return False
        if buried > 0 and dwell_ticks < max_dwell:
            return False
        if active_civilians > 0 and dwell_ticks < max_dwell:
            return False
        if hint_count > 0 and dwell_ticks < max_dwell:
            return False
        return True

    def _finalize_building_if_exhausted(self, building_id: Optional[EntityID], building: Optional[Building], runtime_state, tick: int, *, reason: str, dwell_ticks: Optional[int] = None, hint_count: Optional[int] = None) -> bool:
        if building_id is None or building is None or isinstance(building, Refuge):
            return False
        close_info = self._building_close_info(
            building_id,
            building,
            runtime_state,
            tick,
            dwell_ticks=dwell_ticks,
            hint_count=hint_count,
        )
        if not bool(close_info.get('exhausted', False)):
            return False

        newly_visited = runtime_state.mark_visited_building(building_id)
        if newly_visited and runtime_state.deferred_rescue_active and building_id != runtime_state.deferred_source_building:
            runtime_state.deferred_swept_count += 1
        block_ticks = int(setting('search.selection.closed_building_block_ticks', 120))
        runtime_state.block_target('search', building_id, tick + block_ticks)
        if self._result == building_id:
            self._result = None
            runtime_state.set_search_target(None, tick)
        self._decision_logger.debug_event('building_exhausted_closed', {
            'reason': reason,
            'building': entity_value(building_id),
            'close_info': close_info,
            'newly_visited': newly_visited,
            'blocked_until': tick + block_ticks,
        })
        return True

    def _looping_on_local_subset(self) -> bool:
        window = int(setting('search.selection.local_loop_target_window', 8))
        unique_limit = int(setting('search.selection.local_loop_unique_target_threshold', 3))
        if len(self._recent_targets) < window:
            return False
        recent = list(self._recent_targets)[-window:]
        return len({entity_value(item) for item in recent}) <= unique_limit

    def _position_loop_active(self, runtime_state) -> bool:
        positions = [entity_value(p) for p in list(getattr(runtime_state, 'recent_positions', [])) if p is not None]
        if len(positions) < 6:
            return False
        recent = positions[-6:]
        return len(set(recent)) <= 3

    def _log_ml_bridge_status_once(self, tick: int) -> None:
        if self._ml_status_logged:
            return
        if tick > int(setting('search.ml_v2.debug_log_until_tick', 3)):
            return
        info = self._ml.describe()
        info['module'] = 'search'
        info['patch_marker'] = 'ml_search'
        self._decision_logger.debug_event('ml_bridge_status', info)
        self._decision_logger.log_text('ML bridge status', info)
        self._ml_status_logged = True

    def update_info(self, message_manager: MessageManager) -> Search:
        super().update_info(message_manager)
        tick = int(self._agent_info.get_time())
        self._log_ml_bridge_status_once(tick)
        self._ensure_static_map_cache()
        self._begin_tick_cache(tick)
        current = self._agent_info.get_position_entity_id()
        current_entity = self._world_info.get_entity(current) if current is not None else None
        runtime_state = get_runtime_state(self._agent_info)
        previous_search_building = runtime_state.search_building_current
        if previous_search_building is not None and previous_search_building != current:
            previous_entity = self._world_info.get_entity(previous_search_building)
            if isinstance(previous_entity, Building) and not isinstance(previous_entity, Refuge):
                previous_dwell_ticks = runtime_state.search_building_dwell_ticks(previous_search_building, tick)
                previous_hint_count = self._hint_count(previous_search_building, fast_only=self._is_large_map)
                self._finalize_building_if_exhausted(
                    previous_search_building,
                    previous_entity,
                    runtime_state,
                    tick,
                    reason='building_transition',
                    dwell_ticks=previous_dwell_ticks,
                    hint_count=previous_hint_count,
                )
            runtime_state.leave_search_building()

        if isinstance(current_entity, Building) and not isinstance(current_entity, Refuge):
            hint_count = self._hint_count(current, fast_only=self._is_large_map)
            dwell_ticks = runtime_state.enter_search_building(current, tick, hint_count)
            required_dwell = self._required_building_dwell(current, current_entity, hint_count)
            if self._should_close_building(current, dwell_ticks, required_dwell, runtime_state, tick):
                closed = self._finalize_building_if_exhausted(
                    current,
                    current_entity,
                    runtime_state,
                    tick,
                    reason='inside_building',
                    dwell_ticks=dwell_ticks,
                    hint_count=hint_count,
                )
                if closed and self._result == current:
                    self._decision_logger.log_text('Здание обследовано полностью', {'цель': entity_value(current), 'dwell_ticks': dwell_ticks, 'required_dwell': required_dwell})
                    self._result = None
                    runtime_state.set_search_target(None, tick)
            else:
                active_inside, buried_inside = self._known_problem_civilians(current)
                if active_inside > 0 or buried_inside > 0 or dwell_ticks < required_dwell:
                    self._decision_logger.debug_event('building_hold', {'building': entity_value(current), 'dwell_ticks': dwell_ticks, 'required_dwell': required_dwell, 'active_inside': active_inside, 'buried_inside': buried_inside, 'hint_count': hint_count})
        else:
            runtime_state.leave_search_building()
            if isinstance(current_entity, Refuge) and self._result == current and runtime_state.detector_target is None:
                runtime_state.block_target('search', current, tick + int(setting('search.selection.current_area_block_ticks', 16)))
                self._result = None
                runtime_state.set_search_target(None, tick)
        return self

    def _detect_two_point_loop(self, runtime_state) -> tuple[Optional[EntityID], Optional[EntityID]]:
        positions = list(getattr(runtime_state, 'recent_positions', []))
        if len(positions) < 4:
            return None, None
        a, b, c, d = positions[-4:]
        if a is None or b is None or c is None or d is None:
            return None, None
        if a == c and b == d and a != b:
            return a, b
        return None, None

    def _can_hold_in_transit(self, runtime_state, tick: int, current_area: Optional[EntityID], current_entity) -> bool:
        if self._startup_launch_enabled(runtime_state, tick):
            return False
        if runtime_state.detector_target is not None:
            return False
        if self._result is None or runtime_state.search_target != self._result:
            return False
        if current_area is None or self._result == current_area:
            return False
        if not isinstance(current_entity, Road):
            return False
        current_age = tick - runtime_state.search_target_since if runtime_state.search_target_since >= 0 else 0
        if current_age > int(setting('search.selection.transit_keep_ticks', 10)):
            return False
        if runtime_state.stationary_ticks > int(setting('search.selection.transit_keep_max_stationary', 1)):
            return False
        if runtime_state.is_blocked('search', self._result, tick):
            return False
        path = self._path_planning.get_path(self._agent_info.get_position_entity_id(), self._result)
        return bool(path)

    def _should_lock_current_building(self, runtime_state, current_area: Optional[EntityID], current_entity, tick: int) -> tuple[bool, dict[str, object]]:
        info: dict[str, object] = {
            'building': entity_value(current_area),
            'dwell_ticks': 0,
            'required_dwell': 0,
            'hint_count': 0,
            'active_inside': 0,
            'buried_inside': 0,
        }
        if runtime_state.detector_target is not None:
            return False, info
        startup_hold_disabled = bool(setting('search.startup.disable_current_building_hold_until_first_real_move', True))
        if startup_hold_disabled and not runtime_state.has_real_movement():
            info['startup_hold_suppressed'] = True
            info['startup_hold_reason'] = 'awaiting_first_real_move'
            return False, info
        if current_area is None or not isinstance(current_entity, Building) or isinstance(current_entity, Refuge):
            return False, info
        if not is_search_target_valid(self._world_info, current_area):
            info['startup_hold_suppressed'] = True
            info['startup_hold_reason'] = 'non_searchable_building'
            return False, info
        if runtime_state.is_blocked('search', current_area, tick):
            return False, info

        hint_count = self._hint_count(current_area, fast_only=self._is_large_map)
        dwell_ticks = runtime_state.search_building_dwell_ticks(current_area, tick)
        required_dwell = self._required_building_dwell(current_area, current_entity, hint_count)
        active_inside, buried_inside = self._known_problem_civilians(current_area)
        info.update({
            'dwell_ticks': dwell_ticks,
            'required_dwell': required_dwell,
            'hint_count': hint_count,
            'active_inside': active_inside,
            'buried_inside': buried_inside,
        })

        if dwell_ticks < required_dwell:
            return True, info
        if buried_inside > 0:
            return True, info
        if active_inside > 0:
            return True, info
        if hint_count > 0 and dwell_ticks < int(setting('search.inspection.max_dwell_ticks', 8)):
            return True, info
        return False, info

    def _partition_candidates(self, runtime_state) -> tuple[set[EntityID], dict[str, list[Building]]]:
        self._ensure_static_map_cache()
        all_buildings = self._static_all_buildings or []
        cluster_ids = self._static_cluster_ids or set()
        pools: dict[str, list[Building]] = {
            'cluster_unvisited': [],
            'outside_unvisited': [],
            'cluster_revisit': [],
            'outside_revisit': [],
        }
        for building in all_buildings:
            building_id = building.get_entity_id()
            if building_id is None:
                continue
            in_cluster = building_id in cluster_ids
            visited = self._is_visited_building(runtime_state, building_id)
            if in_cluster and not visited:
                pools['cluster_unvisited'].append(building)
            elif in_cluster and visited:
                pools['cluster_revisit'].append(building)
            elif not in_cluster and not visited:
                pools['outside_unvisited'].append(building)
            else:
                pools['outside_revisit'].append(building)
        return cluster_ids, pools

    def _choose_scope(self, pools: dict[str, list[Building]], known_civilians_count: int, tick: int, runtime_state) -> tuple[str, list[Building], str, bool, float]:
        cluster_total = len(pools['cluster_unvisited']) + len(pools['cluster_revisit'])
        cluster_remaining_ratio = 0.0 if cluster_total <= 0 else len(pools['cluster_unvisited']) / float(cluster_total)
        exhaustion_ratio = float(setting('search.selection.cluster_exhaustion_ratio', 0.15))
        stagnation_window = int(setting('search.selection.stagnation_window', 6))
        stagnation_active = len(self._recent_progress) >= stagnation_window and not any(self._recent_progress)
        local_loop_active = self._looping_on_local_subset() or self._position_loop_active(runtime_state)
        escape_ticks = int(setting('search.selection.global_escape_ticks', 8))
        revisit_budget = int(setting('search.selection.cluster_revisit_budget', 3))
        no_known_civilians = known_civilians_count <= 0

        if pools['outside_unvisited'] and self._forced_global_until >= tick:
            return 'outside_unvisited', pools['outside_unvisited'], 'forced_global_cooldown', True, cluster_remaining_ratio

        if pools['cluster_unvisited']:
            if pools['outside_unvisited'] and (stagnation_active or local_loop_active):
                self._forced_global_until = tick + escape_ticks
                return 'outside_unvisited', pools['outside_unvisited'], 'stagnation_escape_global', True, cluster_remaining_ratio
            if pools['outside_unvisited'] and cluster_remaining_ratio <= exhaustion_ratio:
                self._forced_global_until = tick + escape_ticks
                return 'outside_unvisited', pools['outside_unvisited'], 'cluster_frontier_escape', True, cluster_remaining_ratio
            if self._is_large_map and pools['outside_unvisited'] and no_known_civilians and stagnation_active:
                self._forced_global_until = tick + escape_ticks
                return 'outside_unvisited', pools['outside_unvisited'], 'large_map_escape_cluster', True, cluster_remaining_ratio
            return 'cluster_unvisited', pools['cluster_unvisited'], 'cluster_priority', False, cluster_remaining_ratio
        if pools['outside_unvisited']:
            return 'outside_unvisited', pools['outside_unvisited'], 'cluster_exhausted', True, cluster_remaining_ratio
        if pools['cluster_revisit']:
            if pools['outside_revisit'] and self._cluster_revisit_streak >= revisit_budget:
                self._forced_global_until = tick + escape_ticks
                return 'outside_revisit', pools['outside_revisit'], 'cluster_revisit_budget_exceeded', True, cluster_remaining_ratio
            return 'cluster_revisit', pools['cluster_revisit'], 'cluster_revisit', False, cluster_remaining_ratio
        if pools['outside_revisit']:
            return 'outside_revisit', pools['outside_revisit'], 'global_revisit', True, cluster_remaining_ratio
        return 'none', [], 'no_candidates', False, cluster_remaining_ratio

    def _prefilter_candidates_for_large_map(self, candidates: list[Building], runtime_state, tick: int) -> list[Building]:
        if not self._is_large_map:
            return candidates
        current_area = self._agent_info.get_position_entity_id()
        if tick <= 1:
            self._decision_logger.debug_event('ml_v2_stage2_fix_active', {'module': 'search', 'fix_version': 'stable_release'})
        if current_area is None or len(candidates) <= int(setting('search.selection.large_map_local_max_candidates', 24)):
            return candidates

        exclude_current_area = bool(setting('search.selection.large_map_exclude_current_area', True))
        blocked_exit_count = runtime_state.count_blocked_first_hops_from(current_area, tick)
        in_recovery = runtime_state.stationary_ticks >= int(setting('search.selection.stationary_release_ticks', 4)) or blocked_exit_count > 0
        sample_size = int(setting('search.selection.large_map_distance_sample', 80))
        local_radius = float(setting('search.selection.large_map_local_radius', 18000.0))
        max_candidates = int(setting('search.selection.large_map_local_max_candidates', 24))
        recovery_max = int(setting('search.selection.large_map_recovery_max_candidates', 36))
        limit = recovery_max if in_recovery else max_candidates

        approx_sorted = sorted(candidates, key=lambda b: self._approx_distance_to(b.get_entity_id()))
        nearest_sample = approx_sorted[: max(sample_size, limit)]
        local = [b for b in nearest_sample if self._approx_distance_to(b.get_entity_id()) <= local_radius]
        if len(local) < limit:
            local = nearest_sample[: max(limit, len(local))]

        hinted = [b for b in local if self._hint_count(b.get_entity_id(), fast_only=True) > 0]
        unvisited = [b for b in local if not self._is_visited_building(runtime_state, b.get_entity_id())]
        ordered_pool: list[Building] = []
        spread_count = int(setting('search.selection.large_map_spread_candidates', 6))
        spread: list[Building] = []
        if approx_sorted:
            step = max(1, len(approx_sorted) // max(spread_count, 1))
            for index in range(0, len(approx_sorted), step):
                spread.append(approx_sorted[index])
                if len(spread) >= spread_count:
                    break
            if approx_sorted[-1] not in spread and len(spread) < spread_count + 1:
                spread.append(approx_sorted[-1])
        ordered_pool.extend(hinted)
        ordered_pool.extend(unvisited)
        ordered_pool.extend(local)
        ordered_pool.extend(spread)
        if in_recovery:
            ordered_pool.extend(approx_sorted[: max(sample_size + 24, limit * 3)])

        result: list[Building] = []
        seen: set[str] = set()
        for building in ordered_pool:
            building_id = building.get_entity_id()
            key = entity_value(building_id)
            if key in seen:
                continue
            if exclude_current_area and building_id == current_area and len(ordered_pool) > 1:
                continue
            seen.add(key)
            result.append(building)
            if len(result) >= limit:
                break
        return result or candidates[:limit]

    def _candidate_features(self, building: Building, cluster_ids: set[EntityID], runtime_state, scope_name: str, tick: int) -> dict[str, float | int | str | bool | None]:
        building_id = building.get_entity_id()
        distance_value = self._distance_cached(self._agent_info.get_entity_id(), building_id)
        centrality = self._centrality(building_id)
        hint_count = self._hint_count(building_id, fast_only=self._is_large_map)
        active_inside, buried_inside = self._known_problem_civilians(building_id)
        in_cluster = building_id in cluster_ids
        visited = self._is_visited_building(runtime_state, building_id)
        recent = building_id in self._recent_targets
        current_target = self._result is not None and building_id == self._result
        current_area = self._agent_info.get_position_entity_id()
        if tick <= 1:
            self._decision_logger.debug_event('ml_v2_stage2_fix_active', {'module': 'search', 'fix_version': 'stable_release'})
        blocked = runtime_state.is_blocked('search', building_id, int(self._agent_info.get_time()))

        heuristic_score = 0.0
        if scope_name.startswith('cluster') and in_cluster:
            heuristic_score += float(setting('search.weights.cluster_bonus', 9000.0))
        if scope_name.startswith('outside') and not in_cluster:
            heuristic_score += float(setting('search.weights.global_exploration_bonus', 9500.0))
        if not visited:
            heuristic_score += float(setting('search.weights.unvisited_bonus', 7000.0))
        heuristic_score += float(setting('search.weights.civilian_hint_bonus', 5000.0)) * hint_count
        heuristic_score += float(setting('search.weights.current_building_civilian_bonus', 3500.0)) * active_inside
        heuristic_score += float(setting('search.weights.current_building_buried_bonus', 4500.0)) * buried_inside
        heuristic_score += float(setting('search.weights.centrality_weight', 1200.0)) * centrality
        heuristic_score -= float(setting('search.weights.distance_weight', 0.22)) * distance_value
        if visited:
            heuristic_score -= float(setting('search.weights.revisit_penalty', 12000.0))
        if recent:
            heuristic_score -= float(setting('search.weights.recent_target_penalty', 4000.0))
        if current_target:
            heuristic_score += float(setting('search.weights.current_target_bonus', 3500.0))
        if blocked:
            heuristic_score -= float(setting('search.weights.blocked_penalty', 50000.0))
        if building_id == current_area and runtime_state.detector_target is None:
            penalty = float(setting('search.weights.current_area_reselect_penalty', 60000.0))
            heuristic_score -= penalty if (visited and hint_count <= 0 and active_inside <= 0) else penalty * 0.25

        return {
            'candidate_id': entity_value(building_id),
            'distance': round(distance_value, 2),
            'centrality': centrality,
            'civilian_hint_count': hint_count,
            'active_civilians_inside': active_inside,
            'buried_inside': buried_inside,
            'in_cluster': in_cluster,
            'visited': visited,
            'recent_target': recent,
            'current_target': current_target,
            'blocked': blocked,
            'scope': scope_name,
            'heuristic_score': round(heuristic_score, 4),
        }

    def _select_candidate_entity(self, candidates: list[Building], candidate_payload: list[dict[str, object]], runtime_state, tick: int) -> tuple[Optional[EntityID], str]:
        by_id = {entity_value(building.get_entity_id()): building.get_entity_id() for building in candidates if building.get_entity_id() is not None}
        top_k = max(1, int(setting('search.selection.path_check_top_k', 6)))
        chosen: Optional[EntityID] = None
        chosen_reason = 'best_score'
        for payload in candidate_payload[:top_k]:
            candidate_id = by_id.get(str(payload['candidate_id']))
            if candidate_id is None:
                continue
            path = self._path_planning.get_path(self._agent_info.get_position_entity_id(), candidate_id)
            payload['path_nodes'] = len(path)
            payload['reachable'] = bool(path)
            payload['first_hop'] = entity_value(path[1] if len(path) >= 2 else None)
            if path:
                chosen = candidate_id
                break
            runtime_state.block_target('search', candidate_id, tick + int(setting('search.selection.unreachable_block_ticks', 18)))
        return chosen, chosen_reason if chosen is not None else 'no_reachable_candidate'

    def _ensure_search_payload_reachability(self, payload: Optional[dict[str, object]], candidates: list[Building], runtime_state, tick: int) -> bool:
        if payload is None:
            return False
        reachable = payload.get('reachable')
        if reachable is not None:
            return bool(reachable)
        by_id = {entity_value(building.get_entity_id()): building.get_entity_id() for building in candidates if building.get_entity_id() is not None}
        candidate_id = by_id.get(str(payload.get('candidate_id')))
        if candidate_id is None:
            payload['path_nodes'] = 0
            payload['reachable'] = False
            payload['first_hop'] = None
            return False
        path = self._path_planning.get_path(self._agent_info.get_position_entity_id(), candidate_id)
        payload['path_nodes'] = len(path)
        payload['reachable'] = bool(path)
        payload['first_hop'] = entity_value(path[1] if len(path) >= 2 else None)
        if not path:
            runtime_state.block_target('search', candidate_id, tick + int(setting('search.selection.unreachable_block_ticks', 18)))
        return bool(path)

    def _ml_override_search_order(
        self,
        candidates: list[Building],
        candidate_payload: list[dict[str, object]],
        heuristic_order: list[dict[str, object]],
        heuristic_best_payload: Optional[dict[str, object]],
        ml_best_payload: Optional[dict[str, object]],
        runtime_state,
        tick: int,
        ml_primary_active: bool,
    ) -> tuple[list[dict[str, object]], str, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            'blend_best_id': str(candidate_payload[0].get('candidate_id')) if candidate_payload else None,
            'override_delta': None,
            'override_threshold': float(setting('search.ml_primary.override_min_delta', 0.015)),
            'override_reachable_required': bool(setting('search.ml_primary.override_only_if_reachable', True)),
            'override_ml_reachable': None,
            'override_heuristic_reachable': None,
            'override_blocked_reason': None,
        }
        if not heuristic_order:
            return heuristic_order, 'heuristic', diagnostics
        if not ml_primary_active or heuristic_best_payload is None or ml_best_payload is None:
            diagnostics['override_blocked_reason'] = 'ml_inactive_or_missing'
            return heuristic_order, 'heuristic', diagnostics

        heuristic_best_id = str(heuristic_best_payload.get('candidate_id'))
        ml_best_id = str(ml_best_payload.get('candidate_id'))
        if ml_best_id == heuristic_best_id:
            diagnostics['override_blocked_reason'] = 'same_winner'
            return heuristic_order, 'heuristic', diagnostics

        heuristic_ml = heuristic_best_payload.get('ml_score')
        ml_best_score = ml_best_payload.get('ml_score')
        if heuristic_ml is None or ml_best_score is None:
            diagnostics['override_blocked_reason'] = 'missing_ml_score'
            return heuristic_order, 'heuristic', diagnostics

        override_delta = float(ml_best_score) - float(heuristic_ml)
        diagnostics['override_delta'] = round(override_delta, 6)
        if override_delta < float(diagnostics['override_threshold']):
            diagnostics['override_blocked_reason'] = 'delta_below_threshold'
            return heuristic_order, 'heuristic', diagnostics

        heuristic_reachable = self._ensure_search_payload_reachability(heuristic_best_payload, candidates, runtime_state, tick)
        ml_reachable = self._ensure_search_payload_reachability(ml_best_payload, candidates, runtime_state, tick)
        diagnostics['override_heuristic_reachable'] = heuristic_reachable
        diagnostics['override_ml_reachable'] = ml_reachable
        if bool(diagnostics['override_reachable_required']) and not ml_reachable:
            diagnostics['override_blocked_reason'] = 'ml_unreachable'
            return heuristic_order, 'heuristic', diagnostics

        ordered = [ml_best_payload] + [item for item in heuristic_order if item is not ml_best_payload]
        diagnostics['override_blocked_reason'] = None
        return ordered, 'ml_override', diagnostics

    def _startup_launch_enabled(self, runtime_state, tick: int) -> bool:
        if not bool(setting('search.startup.force_launch_until_first_real_move', True)):
            return False
        max_ticks = int(setting('search.startup.force_launch_max_ticks', 8))
        return runtime_state.startup_launch_active(tick, max_ticks)

    def _startup_launch_select(self, runtime_state, cluster_ids: set[EntityID], current_area: Optional[EntityID], tick: int) -> tuple[Optional[EntityID], str, dict[str, object], list[dict[str, object]]]:
        metadata: dict[str, object] = {
            'startup_launch_active': True,
            'startup_ticks_elapsed': runtime_state.startup_ticks_elapsed(tick),
            'probe_limit': int(setting('search.startup.launch_probe_limit', 16)),
            'candidate_limit': int(setting('search.startup.launch_candidate_limit', 24)),
        }
        preview: list[dict[str, object]] = []
        all_buildings = self._static_all_buildings or []
        cluster_keys = {entity_value(entity_id) for entity_id in cluster_ids if entity_id is not None}
        ranked: list[tuple[tuple[float, float, float, float], Building, dict[str, object]]] = []
        for building in all_buildings:
            building_id = building.get_entity_id()
            if building_id is None:
                continue
            if current_area is not None and building_id == current_area:
                continue
            if not is_search_target_valid(self._world_info, building_id):
                continue
            if runtime_state.is_blocked('search', building_id, tick):
                continue
            hinted = self._hint_count(building_id, fast_only=self._is_large_map)
            in_cluster = entity_value(building_id) in cluster_keys
            visited = self._is_visited_building(runtime_state, building_id)
            approx_distance = self._approx_distance_to(building_id)
            centrality = float(self._centrality(building_id))
            bucket = 6
            if hinted > 0 and not visited and in_cluster:
                bucket = 0
            elif hinted > 0 and not visited:
                bucket = 1
            elif not visited and in_cluster:
                bucket = 2
            elif not visited:
                bucket = 3
            elif hinted > 0:
                bucket = 4
            elif in_cluster:
                bucket = 5
            ranked.append(((float(bucket), approx_distance, -float(hinted), -centrality), building, {
                'candidate_id': entity_value(building_id),
                'startup_bucket': bucket,
                'hint_count': hinted,
                'visited': visited,
                'in_cluster': in_cluster,
                'approx_distance': round(approx_distance, 2),
                'centrality': round(centrality, 2),
            }))
        ranked.sort(key=lambda item: item[0])
        candidate_limit = max(1, int(setting('search.startup.launch_candidate_limit', 24)))
        probe_limit = max(1, int(setting('search.startup.launch_probe_limit', 16)))
        shortlist = ranked[:candidate_limit]
        metadata['shortlist_count'] = len(shortlist)
        chosen: Optional[EntityID] = None
        chosen_reason = 'startup_launch_no_reachable_candidate'
        for _, building, payload in shortlist[:probe_limit]:
            building_id = building.get_entity_id()
            if building_id is None:
                continue
            path = self._path_planning.get_path(self._agent_info.get_position_entity_id(), building_id)
            payload['path_nodes'] = len(path)
            payload['reachable'] = bool(path)
            preview.append(payload)
            if not path:
                continue
            chosen = building_id
            bucket = int(payload['startup_bucket'])
            bucket_reason = {
                0: 'hinted_cluster_unvisited',
                1: 'hinted_unvisited',
                2: 'cluster_unvisited',
                3: 'global_unvisited',
                4: 'hinted_revisit',
                5: 'cluster_revisit',
            }.get(bucket, 'global_revisit')
            chosen_reason = f'startup_launch_{bucket_reason}'
            metadata['selected_bucket'] = bucket
            metadata['selected_hint_count'] = payload['hint_count']
            metadata['selected_in_cluster'] = payload['in_cluster']
            metadata['selected_approx_distance'] = payload['approx_distance']
            metadata['selected_path_nodes'] = len(path)
            break
        if chosen is None:
            preview.extend(payload for _, _, payload in shortlist[probe_limit: min(len(shortlist), probe_limit + 6)])
        return chosen, chosen_reason, metadata, preview

    def _search_target_commitment_enabled(self) -> bool:
        return bool(setting('search.selection.enable_target_commitment', True))

    def _apply_target_commitment(
        self,
        *,
        chosen: Optional[EntityID],
        chosen_reason: str,
        candidate_payload: list[dict[str, object]],
        candidates: list[Building],
        runtime_state,
        tick: int,
        current_area: Optional[EntityID],
        current_entity,
    ) -> tuple[Optional[EntityID], str, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            'commitment_applied': False,
            'commitment_reason': None,
            'commitment_candidate_id': entity_value(chosen),
            'commitment_kept_target_id': entity_value(self._result),
            'commitment_backtrack_detected': False,
            'commitment_flip_detected': False,
            'commitment_current_reachable': None,
            'commitment_candidate_reachable': None,
            'commitment_age': None,
        }
        if not self._search_target_commitment_enabled():
            diagnostics['commitment_reason'] = 'disabled'
            return chosen, chosen_reason, diagnostics
        if runtime_state.detector_target is not None:
            diagnostics['commitment_reason'] = 'detector_active'
            return chosen, chosen_reason, diagnostics
        if self._result is None or runtime_state.search_target != self._result:
            diagnostics['commitment_reason'] = 'no_active_target'
            return chosen, chosen_reason, diagnostics
        if chosen is None or chosen == self._result:
            diagnostics['commitment_reason'] = 'same_target'
            return chosen, chosen_reason, diagnostics
        if current_area is None or not isinstance(current_entity, Road):
            diagnostics['commitment_reason'] = 'not_in_transit'
            return chosen, chosen_reason, diagnostics
        if not is_search_target_valid(self._world_info, self._result):
            diagnostics['commitment_reason'] = 'current_target_invalid'
            return chosen, chosen_reason, diagnostics
        current_age = tick - runtime_state.search_target_since if runtime_state.search_target_since >= 0 else 0
        diagnostics['commitment_age'] = current_age
        max_age = int(setting('search.selection.target_commitment_ticks', 12))
        if current_age > max_age:
            diagnostics['commitment_reason'] = 'target_too_old'
            return chosen, chosen_reason, diagnostics

        current_payload = next((item for item in candidate_payload if str(item.get('candidate_id')) == entity_value(self._result)), None)
        candidate_item = next((item for item in candidate_payload if str(item.get('candidate_id')) == entity_value(chosen)), None)
        if current_payload is None or candidate_item is None:
            diagnostics['commitment_reason'] = 'payload_missing'
            return chosen, chosen_reason, diagnostics

        current_reachable = self._ensure_search_payload_reachability(current_payload, candidates, runtime_state, tick)
        candidate_reachable = self._ensure_search_payload_reachability(candidate_item, candidates, runtime_state, tick)
        diagnostics['commitment_current_reachable'] = current_reachable
        diagnostics['commitment_candidate_reachable'] = candidate_reachable
        if not current_reachable:
            diagnostics['commitment_reason'] = 'current_unreachable'
            return chosen, chosen_reason, diagnostics
        if not candidate_reachable:
            diagnostics['commitment_reason'] = 'candidate_unreachable'
            return chosen, chosen_reason, diagnostics

        recent_targets = list(self._recent_targets)
        flip_detected = (
            len(recent_targets) >= 2
            and recent_targets[-1] == self._result
            and recent_targets[-2] == chosen
            and chosen != self._result
        )
        diagnostics['commitment_flip_detected'] = flip_detected

        previous_position = runtime_state.previous_position
        candidate_first_hop = candidate_item.get('first_hop')
        current_first_hop = current_payload.get('first_hop')
        backtrack_detected = (
            previous_position is not None
            and entity_value(previous_position) != 'None'
            and str(candidate_first_hop) == entity_value(previous_position)
            and str(current_first_hop) != entity_value(previous_position)
        )
        diagnostics['commitment_backtrack_detected'] = backtrack_detected

        score_margin = float(setting('search.selection.target_commitment_score_margin', 2500.0))
        candidate_score = float(candidate_item.get('final_score') or candidate_item.get('heuristic_score') or 0.0)
        current_score = float(current_payload.get('final_score') or current_payload.get('heuristic_score') or 0.0)
        score_gap = candidate_score - current_score
        diagnostics['commitment_score_gap'] = round(score_gap, 6)

        current_scope = str(current_payload.get('scope') or '')
        candidate_scope = str(candidate_item.get('scope') or '')
        commit_unvisited_only = bool(setting('search.selection.target_commitment_unvisited_only', True))
        if commit_unvisited_only and bool(current_payload.get('visited')):
            diagnostics['commitment_reason'] = 'current_target_already_entered'
            return chosen, chosen_reason, diagnostics
        if current_scope.endswith('revisit') and candidate_scope.endswith('unvisited') and score_gap > 0.0:
            diagnostics['commitment_reason'] = 'prefer_unvisited_upgrade'
            return chosen, chosen_reason, diagnostics

        if not flip_detected and not backtrack_detected and score_gap > score_margin:
            diagnostics['commitment_reason'] = 'candidate_gain_too_large'
            return chosen, chosen_reason, diagnostics

        if flip_detected:
            block_ticks = int(setting('search.selection.flip_flop_block_ticks', 10))
            runtime_state.block_target('search', chosen, tick + block_ticks)
            diagnostics['flip_blocked_until'] = tick + block_ticks

        diagnostics['commitment_applied'] = True
        diagnostics['commitment_reason'] = 'flip_prevented' if flip_detected else 'backtrack_prevented' if backtrack_detected else 'target_commitment'
        return self._result, 'target_commitment', diagnostics

    def _search_ml_primary_active(self) -> bool:
        return self._ml.mode_name() in ('hybrid', 'pure_ml_test') and bool(setting('search.ml_primary.enabled', True))

    def _allow_building_hold_override(self, runtime_state, hold_info: dict[str, object]) -> bool:
        if not self._search_ml_primary_active():
            return False
        if not bool(setting('search.ml_primary.allow_building_hold_override', True)):
            return False
        if not runtime_state.has_real_movement():
            return False
        if bool(hold_info.get('active_inside')) or bool(hold_info.get('buried_inside')):
            return False
        hint_count = int(hold_info.get('hint_count') or 0)
        if hint_count > int(setting('search.ml_primary.max_hint_count_for_hold_override', 1)):
            return False
        dwell_ticks = int(hold_info.get('dwell_ticks') or 0)
        required_dwell = int(hold_info.get('required_dwell') or 0)
        dwell_deficit = max(0, required_dwell - dwell_ticks)
        if dwell_deficit > int(setting('search.ml_primary.max_dwell_deficit_for_hold_override', 1)):
            return False
        return True

    def _allow_transit_keep_override(self, runtime_state) -> bool:
        if not self._search_ml_primary_active():
            return False
        if not bool(setting('search.ml_primary.allow_transit_keep_override', False)):
            return False
        if not runtime_state.has_real_movement():
            return False
        if runtime_state.stationary_ticks > int(setting('search.ml_primary.max_stationary_for_transit_override', 1)):
            return False
        return True

    def calculate(self) -> Search:
        started = time.monotonic()
        tick = int(self._agent_info.get_time())
        self._ensure_static_map_cache()
        self._begin_tick_cache(tick)
        runtime_state = get_runtime_state(self._agent_info)
        runtime_state.cleanup(tick)
        runtime_state.update_position(self._agent_info.get_position_entity_id(), tick)
        if tick <= int(setting('search.startup.debug_ticks', 20)):
            self._decision_logger.debug_event(
                'startup_motion_state',
                {
                    'tick': tick,
                    'position': entity_value(self._agent_info.get_position_entity_id()),
                    'startup_position': entity_value(runtime_state.startup_position),
                    'first_real_move_tick': runtime_state.first_real_move_tick,
                    'has_real_movement': runtime_state.has_real_movement(),
                    'stationary_ticks': runtime_state.stationary_ticks,
                },
            )
        for search_outcome in runtime_state.collect_search_outcomes(tick):
            self._decision_logger.state_snapshot('search_selection_outcome', search_outcome)
            self._decision_logger.debug_event('search_selection_outcome', search_outcome)
        current_area = self._agent_info.get_position_entity_id()
        if tick <= 1:
            self._decision_logger.debug_event('ml_v2_stage2_fix_active', {'module': 'search', 'fix_version': 'stable_release'})
        current_entity = self._world_info.get_entity(current_area) if current_area is not None else None

        if runtime_state.detector_target is None and current_area is not None and runtime_state.previous_position is not None and current_area != runtime_state.previous_position and isinstance(current_entity, Road):
            runtime_state.block_target('search', runtime_state.previous_position, tick + int(setting('search.selection.backtrack_cooldown_ticks', 6)))

        loop_a, loop_b = self._detect_two_point_loop(runtime_state)
        if runtime_state.detector_target is None and current_area is not None and loop_a is not None and loop_b is not None:
            if current_area == loop_a:
                runtime_state.block_target('search', loop_b, tick + int(setting('search.selection.oscillation_block_ticks', 12)))
            elif current_area == loop_b:
                runtime_state.block_target('search', loop_a, tick + int(setting('search.selection.oscillation_block_ticks', 12)))

        runtime_state.carrying = self._agent_info.some_one_on_board() is not None
        if runtime_state.carrying:
            runtime_state.phase = 'transport'
        elif runtime_state.detector_target is not None:
            runtime_state.phase = 'move_to_victim'
        else:
            runtime_state.phase = 'search'

        if runtime_state.detector_target is None and self._startup_launch_enabled(runtime_state, tick):
            cluster_ids = self._static_cluster_ids or set()
            chosen, chosen_reason, startup_launch_info, startup_preview = self._startup_launch_select(runtime_state, cluster_ids, current_area, tick)
            self._decision_logger.debug_event(
                'startup_launch_probe',
                {
                    'tick': tick,
                    'current_area': entity_value(current_area),
                    'chosen': entity_value(chosen),
                    'chosen_reason': chosen_reason,
                    'startup_ticks_elapsed': runtime_state.startup_ticks_elapsed(tick),
                    'preview_count': len(startup_preview),
                    'first_real_move_tick': runtime_state.first_real_move_tick,
                },
            )
            if chosen is not None:
                self._result = chosen
                runtime_state.set_search_target(chosen, tick)
                if not self._recent_targets or self._recent_targets[-1] != chosen:
                    self._recent_targets.append(chosen)
                self._recent_progress.append(True)
                compute_ms = round((time.monotonic() - started) * 1000.0, 3)
                metadata = {
                    'selection_mode': 'startup_launch',
                    'selected_by': 'heuristic',
                    'exploration_used': False,
                    'selected_reason': chosen_reason,
                    'search_scope': 'startup_launch',
                    'top_k_candidates': [item.get('candidate_id') for item in startup_preview[:8]],
                    'selected_rank_by_heuristic': 1,
                    'selected_rank_by_ml': None,
                    'selected_rank_by_final': 1,
                    'startup_launch': startup_launch_info,
                }
                self._decision_logger.decision_snapshot(
                    'search',
                    {
                        'current_area': entity_value(current_area),
                        'cluster_candidate_count': len(self._static_cluster_buildings or []),
                        'global_candidate_count': self._all_buildings_count,
                        'visited_count': self._visited_count(runtime_state),
                        'local_loop_active': self._looping_on_local_subset(),
                        'position_loop_active': self._position_loop_active(runtime_state),
                        'cluster_revisit_streak': self._cluster_revisit_streak,
                        'phase': runtime_state.phase,
                        'active_detector_target': entity_value(runtime_state.detector_target),
                        'search_mode': self._ml.mode_name(),
                        'previous_position': entity_value(runtime_state.previous_position),
                        'loop_a': entity_value(loop_a),
                        'loop_b': entity_value(loop_b),
                        'startup_launch': startup_launch_info,
                    },
                    startup_preview,
                    entity_value(chosen),
                    chosen_reason,
                    metadata=metadata,
                )
                runtime_state.register_search_selection(
                    chosen,
                    tick,
                    int(setting('search.ml_v2.outcome_window_ticks', 10)),
                    selected_rank=1,
                    ml_rank=-1,
                    final_rank=1,
                    selected_by='heuristic',
                    selection_mode='startup_launch',
                    reason=chosen_reason,
                    scope='startup_launch',
                    exploration_used=False,
                    candidate_count=len(startup_preview),
                    top_k_candidates=[item.get('candidate_id') for item in startup_preview[:8]],
                )
                self._decision_logger.log_text(
                    'Выбрана следующая зона поиска',
                    {
                        'цель': entity_value(chosen),
                        'причина': chosen_reason,
                        'фаза': runtime_state.phase,
                        'startup_ticks_elapsed': runtime_state.startup_ticks_elapsed(tick),
                        'selected_path_nodes': startup_launch_info.get('selected_path_nodes'),
                        'previous_position': entity_value(runtime_state.previous_position),
                    },
                )
                self._decision_logger.state_snapshot('agent_runtime_state', runtime_state.snapshot())
                self._decision_logger.debug_event(
                    'search_cycle_summary',
                    {
                        'compute_ms': compute_ms,
                        'timings_ms': {},
                        'chosen': entity_value(chosen),
                        'chosen_reason': chosen_reason,
                        'scope_name': 'startup_launch',
                        'scope_reason': chosen_reason,
                        'candidate_count': len(startup_preview),
                        'stationary_ticks': runtime_state.stationary_ticks,
                        'blocked_first_hops_from_current': runtime_state.blocked_first_hops_from(current_area, tick),
                        'current_target': entity_value(self._result),
                        'previous_position': entity_value(runtime_state.previous_position),
                        'loop_a': entity_value(loop_a),
                        'loop_b': entity_value(loop_b),
                        'startup_launch': startup_launch_info,
                    },
                )
                return self

        hold_current_building, hold_info = self._should_lock_current_building(runtime_state, current_area, current_entity, tick)
        hold_override_active = hold_current_building and self._allow_building_hold_override(runtime_state, hold_info)
        if hold_override_active:
            self._decision_logger.debug_event('search_hold_override', {'tick': tick, 'current_area': entity_value(current_area), 'hold_info': hold_info})
        if hold_current_building and not hold_override_active:
            chosen = current_area
            self._result = chosen
            runtime_state.set_search_target(chosen, tick)
            if chosen is not None and (not self._recent_targets or self._recent_targets[-1] != chosen):
                self._recent_targets.append(chosen)
            self._recent_progress.append(True)
            compute_ms = round((time.monotonic() - started) * 1000.0, 3)
            hold_metadata = {
                'selection_mode': self._ml.mode_name(),
                'selected_by': 'heuristic',
                'exploration_used': False,
                'selected_reason': 'building_hold_lock',
                'search_scope': 'current_building',
                'top_k_candidates': [entity_value(chosen)] if chosen is not None else [],
                'selected_rank_by_heuristic': 1 if chosen is not None else None,
                'selected_rank_by_ml': None,
                'selected_rank_by_final': 1 if chosen is not None else None,
            }
            self._decision_logger.decision_snapshot(
                'search',
                {
                    'current_area': entity_value(current_area),
                    'cluster_candidate_count': len(self._static_cluster_buildings or []),
                    'global_candidate_count': self._all_buildings_count,
                    'visited_count': self._visited_count(runtime_state),
                    'local_loop_active': self._looping_on_local_subset(),
                    'position_loop_active': self._position_loop_active(runtime_state),
                    'cluster_revisit_streak': self._cluster_revisit_streak,
                    'phase': runtime_state.phase,
                    'active_detector_target': entity_value(runtime_state.detector_target),
                    'search_mode': self._ml.mode_name(),
                    'previous_position': entity_value(runtime_state.previous_position),
                    'loop_a': entity_value(loop_a),
                    'loop_b': entity_value(loop_b),
                    'hold_info': hold_info,
                },
                [],
                entity_value(chosen) if chosen is not None else None,
                'building_hold_lock',
                metadata=hold_metadata,
            )
            runtime_state.register_search_selection(
                chosen,
                tick,
                int(setting('search.ml_v2.outcome_window_ticks', 10)),
                selected_rank=1 if chosen is not None else -1,
                ml_rank=-1,
                final_rank=1 if chosen is not None else -1,
                selected_by='heuristic',
                selection_mode=self._ml.mode_name(),
                reason='building_hold_lock',
                scope='current_building',
                exploration_used=False,
                candidate_count=0,
                top_k_candidates=[entity_value(chosen)] if chosen is not None else [],
            )
            self._decision_logger.log_text(
                'Выбрана следующая зона поиска',
                {
                    'цель': entity_value(chosen),
                    'причина': 'building_hold_lock',
                    'фаза': runtime_state.phase,
                    'предыдущая_позиция': entity_value(runtime_state.previous_position),
                    'loop_a': entity_value(loop_a),
                    'loop_b': entity_value(loop_b),
                    'dwell_ticks': hold_info.get('dwell_ticks'),
                    'required_dwell': hold_info.get('required_dwell'),
                    'hint_count': hold_info.get('hint_count'),
                    'active_inside': hold_info.get('active_inside'),
                    'buried_inside': hold_info.get('buried_inside'),
                },
            )
            self._decision_logger.state_snapshot('agent_runtime_state', runtime_state.snapshot())
            self._decision_logger.debug_event(
                'search_cycle_summary',
                {
                    'compute_ms': compute_ms,
                    'timings_ms': {},
                    'chosen': entity_value(chosen),
                    'chosen_reason': 'building_hold_lock',
                    'scope_name': 'current_building',
                    'scope_reason': 'building_hold_lock',
                    'candidate_count': 0,
                    'stationary_ticks': runtime_state.stationary_ticks,
                    'blocked_first_hops_from_current': runtime_state.blocked_first_hops_from(current_area, tick),
                    'current_target': entity_value(self._result),
                    'previous_position': entity_value(runtime_state.previous_position),
                    'loop_a': entity_value(loop_a),
                    'loop_b': entity_value(loop_b),
                    'hold_info': hold_info,
                },
            )
            return self

        transit_keep_active = self._can_hold_in_transit(runtime_state, tick, current_area, current_entity)
        transit_keep_override = transit_keep_active and self._allow_transit_keep_override(runtime_state)
        if transit_keep_override:
            self._decision_logger.debug_event('search_transit_keep_override', {'tick': tick, 'current_area': entity_value(current_area), 'search_target': entity_value(self._result)})
        if transit_keep_active and not transit_keep_override:
            chosen = self._result
            runtime_state.set_search_target(chosen, tick)
            compute_ms = round((time.monotonic() - started) * 1000.0, 3)
            transit_metadata = {
                'selection_mode': self._ml.mode_name(),
                'selected_by': 'heuristic',
                'exploration_used': False,
                'selected_reason': 'transit_keep',
                'search_scope': 'in_transit',
                'top_k_candidates': [entity_value(chosen)] if chosen is not None else [],
                'selected_rank_by_heuristic': 1 if chosen is not None else None,
                'selected_rank_by_ml': None,
                'selected_rank_by_final': 1 if chosen is not None else None,
            }
            self._decision_logger.decision_snapshot(
                'search',
                {
                    'current_area': entity_value(current_area),
                    'cluster_candidate_count': len(self._static_cluster_buildings or []),
                    'global_candidate_count': self._all_buildings_count,
                    'visited_count': self._visited_count(runtime_state),
                'local_loop_active': self._looping_on_local_subset(),
                'position_loop_active': self._position_loop_active(runtime_state),
                'cluster_revisit_streak': self._cluster_revisit_streak,
                    'phase': runtime_state.phase,
                    'active_detector_target': entity_value(runtime_state.detector_target),
                    'search_mode': self._ml.mode_name(),
                    'previous_position': entity_value(runtime_state.previous_position),
                    'loop_a': entity_value(loop_a),
                    'loop_b': entity_value(loop_b),
                },
                [],
                entity_value(chosen) if chosen is not None else None,
                'transit_keep',
                metadata=transit_metadata,
            )
            runtime_state.register_search_selection(
                chosen,
                tick,
                int(setting('search.ml_v2.outcome_window_ticks', 10)),
                selected_rank=1 if chosen is not None else -1,
                ml_rank=-1,
                final_rank=1 if chosen is not None else -1,
                selected_by='heuristic',
                selection_mode=self._ml.mode_name(),
                reason='transit_keep',
                scope='in_transit',
                exploration_used=False,
                candidate_count=0,
                top_k_candidates=[entity_value(chosen)] if chosen is not None else [],
            )
            self._decision_logger.log_text(
                'Выбрана следующая зона поиска',
                {
                    'цель': entity_value(chosen),
                    'причина': 'transit_keep',
                    'фаза': runtime_state.phase,
                    'предыдущая_позиция': entity_value(runtime_state.previous_position),
                    'loop_a': entity_value(loop_a),
                    'loop_b': entity_value(loop_b),
                },
            )
            self._decision_logger.state_snapshot('agent_runtime_state', runtime_state.snapshot())
            self._decision_logger.debug_event(
                'search_cycle_summary',
                {
                    'compute_ms': compute_ms,
                    'timings_ms': {},
                    'chosen': entity_value(chosen),
                    'chosen_reason': 'transit_keep',
                    'scope_name': 'hold',
                    'scope_reason': 'transit_keep',
                    'candidate_count': 0,
                    'stationary_ticks': runtime_state.stationary_ticks,
                    'blocked_first_hops_from_current': runtime_state.blocked_first_hops_from(current_area, tick),
                    'current_target': entity_value(self._result),
                'previous_position': entity_value(runtime_state.previous_position),
                'loop_a': entity_value(loop_a),
                'loop_b': entity_value(loop_b),
                    'previous_position': entity_value(runtime_state.previous_position),
                    'loop_a': entity_value(loop_a),
                    'loop_b': entity_value(loop_b),
                },
            )
            return self

        timings: dict[str, float] = {}
        t0 = time.monotonic()
        known_civilians_count = len(self._visible_civilians_cache)
        known_refuges_count = len(refuge_entities(self._world_info))
        timings['world_scan_ms'] = (time.monotonic() - t0) * 1000.0

        t0 = time.monotonic()
        cluster_ids, pools = self._partition_candidates(runtime_state)
        timings['partition_ms'] = (time.monotonic() - t0) * 1000.0

        t0 = time.monotonic()
        scope_name, candidates, scope_reason, forced_global, cluster_remaining_ratio = self._choose_scope(pools, known_civilians_count, tick, runtime_state)
        if scope_name == 'cluster_revisit':
            self._cluster_revisit_streak += 1
        else:
            self._cluster_revisit_streak = 0
        prefilter_before = len(candidates)
        candidates = self._prefilter_candidates_for_large_map(candidates, runtime_state, tick)
        timings['prefilter_ms'] = (time.monotonic() - t0) * 1000.0
        self._decision_logger.debug_event(
            'search_prefilter',
            {
                'scope_name': scope_name,
                'scope_reason': scope_reason,
                'before_count': prefilter_before,
                'after_count': len(candidates),
                'current_area': entity_value(self._agent_info.get_position_entity_id()),
                'stationary_ticks': runtime_state.stationary_ticks,
                'blocked_first_hops_from_current': runtime_state.blocked_first_hops_from(self._agent_info.get_position_entity_id(), tick),
            },
        )

        if not candidates:
            self._result = None
            runtime_state.set_search_target(None, tick)
            self._decision_logger.log_text('Подходящих зданий для поиска нет', {'фаза': runtime_state.phase})
            self._decision_logger.decision_snapshot(
                'search',
                {
                    'current_area': entity_value(self._agent_info.get_position_entity_id()),
                    'visited_count': self._visited_count(runtime_state),
                'local_loop_active': self._looping_on_local_subset(),
                'position_loop_active': self._position_loop_active(runtime_state),
                'cluster_revisit_streak': self._cluster_revisit_streak,
                    'phase': runtime_state.phase,
                    'search_scope': scope_name,
                    'scope_reason': scope_reason,
                },
                [],
                None,
                'no_candidates',
            )
            return self

        t0 = time.monotonic()
        candidate_payload = [self._candidate_features(building, cluster_ids, runtime_state, scope_name, tick) for building in candidates if building.get_entity_id() is not None]
        candidate_payload.sort(key=lambda item: float(item['heuristic_score']), reverse=True)
        timings['feature_ms'] = (time.monotonic() - t0) * 1000.0

        if isinstance(current_entity, Refuge) and runtime_state.detector_target is None:
            runtime_state.block_target('search', current_area, tick + int(setting('search.selection.current_area_block_ticks', 20)))
        if current_area is not None and runtime_state.detector_target is None:
            filtered_payload = []
            for item in candidate_payload:
                if item.get('candidate_id') == entity_value(current_area) and bool(item.get('visited')) and int(item.get('civilian_hint_count', 0)) <= 0 and int(item.get('active_civilians_inside', 0)) <= 0:
                    runtime_state.block_target('search', current_area, tick + int(setting('search.selection.current_area_block_ticks', 8)))
                    continue
                filtered_payload.append(item)
            if filtered_payload:
                candidate_payload = filtered_payload
        candidate_payload.sort(key=lambda item: float(item['heuristic_score']), reverse=True)
        for index, payload in enumerate(candidate_payload, start=1):
            payload['heuristic_rank'] = index

        rerank_top_k = int(setting('search.ml_v2.rerank_top_k', 8))
        rerank_candidates = candidate_payload[:rerank_top_k] if rerank_top_k > 0 else candidate_payload
        t0 = time.monotonic()
        ml_scores = self._ml.score_candidates(
            {
                'tick': tick,
                'current_area': entity_value(self._agent_info.get_position_entity_id()),
                'visited_count': self._visited_count(runtime_state),
                'local_loop_active': self._looping_on_local_subset(),
                'position_loop_active': self._position_loop_active(runtime_state),
                'cluster_revisit_streak': self._cluster_revisit_streak,
                'candidate_count': len(candidate_payload),
                'rerank_candidate_count': len(rerank_candidates),
                'phase': runtime_state.phase,
                'search_scope': scope_name,
                'scope_reason': scope_reason,
                'cluster_remaining_ratio': round(cluster_remaining_ratio, 4),
                'known_civilians': known_civilians_count,
            },
            rerank_candidates,
        )
        timings['ml_ms'] = (time.monotonic() - t0) * 1000.0

        mode = self._ml.mode_name()
        ml_weight = float(setting('search.ml_primary.ml_alpha', 0.85))
        heuristic_weight = float(setting('search.ml_primary.heuristic_beta', 0.15))
        heuristic_softmax_temp = float(setting('search.ml_primary.heuristic_softmax_temperature', 3.0))
        flat_spread_eps = float(setting('search.ml.flat_score_spread_epsilon', 0.0005))
        flat_margin_eps = float(setting('search.ml.flat_score_top_margin_epsilon', 0.0001))
        fallback_on_low_conf = bool(setting('search.ml_primary.allow_heuristic_fallback', True))
        min_confidence = float(setting('search.ml_primary.min_confidence', 0.50))
        min_margin = float(setting('search.ml_primary.min_margin', 0.002))

        score_values: list[float] = []
        for payload in candidate_payload:
            raw_ml_score = ml_scores.get(str(payload['candidate_id']))
            payload['ml_score'] = round(float(raw_ml_score), 6) if raw_ml_score is not None else None
            if raw_ml_score is not None:
                score_values.append(float(raw_ml_score))

        ml_ranked = [item for item in candidate_payload if item.get('ml_score') is not None]
        ml_ranked.sort(key=lambda item: float(item.get('ml_score') or 0.0), reverse=True)
        for index, payload in enumerate(ml_ranked, start=1):
            payload['ml_rank'] = index

        heuristic_values = [float(item.get('heuristic_score') or 0.0) for item in candidate_payload]
        if candidate_payload:
            if len(candidate_payload) == 1:
                heuristic_components = [1.0]
            else:
                h_max = max(heuristic_values)
                h_min = min(heuristic_values)
                h_span = max(h_max - h_min, 1.0)
                logits = [((value - h_max) / h_span) * heuristic_softmax_temp for value in heuristic_values]
                max_logit = max(logits)
                exp_values = [math.exp(logit - max_logit) for logit in logits]
                exp_sum = sum(exp_values) or 1.0
                heuristic_components = [value / exp_sum for value in exp_values]
            for payload, component in zip(candidate_payload, heuristic_components):
                payload['heuristic_component'] = round(float(component), 6)
        else:
            heuristic_components = []

        low_confidence_ml = False
        ml_score_spread = None
        ml_top_margin = None
        if ml_ranked:
            top_score = float(ml_ranked[0].get('ml_score') or 0.0)
            if len(ml_ranked) >= 2:
                second_score = float(ml_ranked[1].get('ml_score') or 0.0)
                ml_score_spread = top_score - float(ml_ranked[-1].get('ml_score') or 0.0)
                ml_top_margin = top_score - second_score
            else:
                ml_score_spread = 0.0
                ml_top_margin = top_score
            if top_score < min_confidence or ((ml_score_spread is not None and ml_score_spread <= flat_spread_eps) or (ml_top_margin is not None and ml_top_margin <= max(min_margin, flat_margin_eps))):
                low_confidence_ml = True

        ml_primary_active = mode in ('hybrid', 'pure_ml_test') and bool(ml_ranked) and not (fallback_on_low_conf and low_confidence_ml)
        for payload in candidate_payload:
            heuristic_component = float(payload.get('heuristic_component') or 0.0)
            raw_ml_score = payload.get('ml_score')
            payload['ml_component'] = round(float(raw_ml_score), 6) if raw_ml_score is not None else None
            if ml_primary_active and raw_ml_score is not None:
                final_score = (ml_weight * float(raw_ml_score)) + (heuristic_weight * heuristic_component)
            else:
                final_score = heuristic_component
            payload['low_confidence_ml'] = low_confidence_ml
            payload['final_score'] = round(final_score, 6)
        candidate_payload.sort(key=lambda item: float(item['final_score']), reverse=True)
        for index, payload in enumerate(candidate_payload, start=1):
            payload['final_rank'] = index

        heuristic_best_payload = next((item for item in candidate_payload if int(item.get('heuristic_rank') or -1) == 1), None)
        ml_best_payload = ml_ranked[0] if ml_ranked else None
        blended_best_payload = candidate_payload[0] if candidate_payload else None
        heuristic_best_id = str(heuristic_best_payload.get('candidate_id')) if heuristic_best_payload is not None else None
        ml_best_id = str(ml_best_payload.get('candidate_id')) if ml_best_payload is not None else None
        heuristic_gap = None
        h_ranked = sorted(candidate_payload, key=lambda item: float(item.get('heuristic_component') or 0.0), reverse=True)
        if len(h_ranked) >= 2:
            heuristic_gap = float(h_ranked[0].get('heuristic_component') or 0.0) - float(h_ranked[1].get('heuristic_component') or 0.0)
        heuristic_order = sorted(candidate_payload, key=lambda item: int(item.get('heuristic_rank') or 10**9))
        selection_order, selected_by_mode, override_info = self._ml_override_search_order(
            candidates,
            candidate_payload,
            heuristic_order,
            heuristic_best_payload,
            ml_best_payload,
            runtime_state,
            tick,
            ml_primary_active,
        )
        selection_best_payload = selection_order[0] if selection_order else None
        final_best_id = str(selection_best_payload.get('candidate_id')) if selection_best_payload is not None else None
        winner_changed_by_ml = bool(selected_by_mode == 'ml_override' and final_best_id and heuristic_best_id and final_best_id != heuristic_best_id)
        override_applied = winner_changed_by_ml and ml_best_id == final_best_id
        if selected_by_mode != 'ml_override':
            selected_by_mode = 'heuristic_fallback' if (ml_ranked and not ml_primary_active) else 'heuristic'
        ranking_diagnostics = {
            'heuristic_best_id': heuristic_best_id,
            'ml_best_id': ml_best_id,
            'final_best_id': final_best_id,
            'winner_changed_by_ml': winner_changed_by_ml,
            'override_applied': override_applied,
            'ml_gap': None if ml_top_margin is None else round(ml_top_margin, 6),
            'heuristic_gap': None if heuristic_gap is None else round(heuristic_gap, 6),
            **override_info,
        }

        t0 = time.monotonic()
        chosen, chosen_reason = self._select_candidate_entity(candidates, selection_order, runtime_state, tick)
        timings['path_probe_ms'] = (time.monotonic() - t0) * 1000.0
        if chosen is not None and chosen_reason == 'best_score':
            if selected_by_mode == 'ml_override':
                chosen_reason = 'ml_override'
            else:
                chosen_reason = 'heuristic_keep'

        keep_target_ticks = int(setting('search.selection.keep_target_ticks', 8))
        switch_margin = float(setting('search.ml_primary.switch_margin', 0.05)) if self._search_ml_primary_active() else float(setting('search.selection.switch_margin', 6000.0))
        current_best = selection_order[0] if selection_order else None
        current_payload = next((item for item in candidate_payload if bool(item.get('current_target'))), None)
        if self._result is not None and is_search_target_valid(self._world_info, self._result) and current_payload is not None and current_best is not None:
            current_age = tick - runtime_state.search_target_since if runtime_state.search_target_since >= 0 else 0
            score_delta = float(current_best['final_score']) - float(current_payload['final_score'])
            allow_sticky_keep = True
            if self._search_ml_primary_active():
                current_target_id = entity_value(self._result)
                sticky_conflict_margin = float(setting('search.ml_primary.sticky_keep_ml_margin', 0.01))
                if ranking_diagnostics.get('ml_best_id') is not None and current_target_id != ranking_diagnostics.get('ml_best_id'):
                    if ranking_diagnostics.get('ml_gap') is None or float(ranking_diagnostics.get('ml_gap') or 0.0) >= sticky_conflict_margin:
                        allow_sticky_keep = False
                if bool(ranking_diagnostics.get('winner_changed_by_ml')):
                    allow_sticky_keep = False
            if allow_sticky_keep and runtime_state.stationary_ticks < int(setting('search.selection.stationary_release_ticks', 3)) and current_age <= keep_target_ticks and score_delta <= switch_margin:
                chosen = self._result
                chosen_reason = 'sticky_keep'

        chosen, chosen_reason, commitment_info = self._apply_target_commitment(
            chosen=chosen,
            chosen_reason=chosen_reason,
            candidate_payload=candidate_payload,
            candidates=candidates,
            runtime_state=runtime_state,
            tick=tick,
            current_area=current_area,
            current_entity=current_entity,
        )
        ranking_diagnostics.update(commitment_info)

        if runtime_state.stationary_ticks >= int(setting('search.selection.stationary_release_ticks', 4)):
            if self._result is not None:
                runtime_state.block_target('search', self._result, tick + int(setting('search.selection.stationary_block_ticks', 18)))
            alternatives = [b for b in candidates if b.get_entity_id() != self._agent_info.get_position_entity_id() and not runtime_state.is_blocked('search', b.get_entity_id(), tick)]
            alternatives.sort(key=lambda b: (0 if self._hint_count(b.get_entity_id(), fast_only=True) > 0 else 1, self._approx_distance_to(b.get_entity_id())))
            for alt in alternatives[:12]:
                alt_id = alt.get_entity_id()
                if alt_id is None:
                    continue
                path = self._path_planning.get_path(self._agent_info.get_position_entity_id(), alt_id)
                if path:
                    chosen = alt_id
                    chosen_reason = 'stationary_release'
                    break

        if chosen is not None and chosen == self._agent_info.get_position_entity_id() and runtime_state.detector_target is None:
            current_item = next((item for item in candidate_payload if item.get('candidate_id') == entity_value(chosen)), None)
            if current_item is not None and bool(current_item.get('visited')) and int(current_item.get('civilian_hint_count', 0)) <= 0 and int(current_item.get('active_civilians_inside', 0)) <= 0:
                runtime_state.block_target('search', chosen, tick + int(setting('search.selection.current_area_block_ticks', 8)))
                chosen = None
                chosen_reason = 'skip_current_area'

        if chosen is None and candidates:
            alternatives = [b for b in candidates if b.get_entity_id() != self._agent_info.get_position_entity_id() and not runtime_state.is_blocked('search', b.get_entity_id(), tick)]
            alternatives.sort(key=lambda b: (0 if self._hint_count(b.get_entity_id(), fast_only=True) > 0 else 1, self._approx_distance_to(b.get_entity_id())))
            for alt in alternatives[:16]:
                alt_id = alt.get_entity_id()
                if alt_id is None:
                    continue
                path = self._path_planning.get_path(self._agent_info.get_position_entity_id(), alt_id)
                if path:
                    chosen = alt_id
                    chosen_reason = 'nearest_reachable_fallback'
                    break

        if chosen_reason == 'sticky_keep':
            selected_by_mode = 'sticky_keep'
        elif chosen_reason in ('stationary_release', 'nearest_reachable_fallback', 'no_reachable_candidate', 'skip_current_area'):
            selected_by_mode = 'heuristic_fallback'
        selection_metadata = self._selection_metadata(candidate_payload, chosen, mode, chosen_reason, scope_name=scope_name, exploration_used=False, selected_by=selected_by_mode, diagnostics=ranking_diagnostics)
        self._result = chosen
        if chosen is not None:
            self._recent_targets.append(chosen)
            runtime_state.set_search_target(chosen, tick)
        else:
            runtime_state.set_search_target(None, tick)
        runtime_state.register_search_selection(
            chosen,
            tick,
            int(setting('search.ml_v2.outcome_window_ticks', 10)),
            selected_rank=int(selection_metadata.get('selected_rank_by_heuristic') or -1),
            ml_rank=int(selection_metadata.get('selected_rank_by_ml') or -1) if selection_metadata.get('selected_rank_by_ml') is not None else -1,
            final_rank=int(selection_metadata.get('selected_rank_by_final') or -1),
            selected_by=str(selection_metadata.get('selected_by') or 'heuristic'),
            selection_mode=str(selection_metadata.get('selection_mode') or mode),
            reason=str(selection_metadata.get('selected_reason') or chosen_reason),
            scope=str(selection_metadata.get('search_scope') or scope_name),
            exploration_used=bool(selection_metadata.get('exploration_used', False)),
            candidate_count=len(candidate_payload),
            top_k_candidates=list(selection_metadata.get('top_k_candidates') or []),
        )

        selected_payload = next((item for item in candidate_payload if item.get('candidate_id') == entity_value(chosen)), None)
        previous_visited = self._last_visited_count
        visited_count = self._visited_count(runtime_state)
        area_key = entity_value(current_area)
        moved_to_new_area = self._last_area_key is not None and area_key != self._last_area_key
        progress_flag = visited_count > previous_visited or bool(runtime_state.detector_target)
        self._recent_progress.append(progress_flag)
        self._last_visited_count = visited_count
        self._last_area_key = area_key

        top_preview = [
            {
                'id': item.get('candidate_id'),
                'score': item.get('final_score'),
                'distance': item.get('distance'),
                'hint': item.get('civilian_hint_count'),
                'reachable': item.get('reachable'),
                'first_hop': item.get('first_hop'),
            }
            for item in candidate_payload[:5]
        ]
        compute_ms = round((time.monotonic() - started) * 1000.0, 3)

        self._decision_logger.decision_snapshot(
            'search',
            {
                'current_area': entity_value(current_area),
                'cluster_candidate_count': len(self._static_cluster_buildings or []),
                'global_candidate_count': self._all_buildings_count,
                'cluster_unvisited': len(pools['cluster_unvisited']),
                'outside_unvisited': len(pools['outside_unvisited']),
                'cluster_revisit': len(pools['cluster_revisit']),
                'outside_revisit': len(pools['outside_revisit']),
                'cluster_remaining_ratio': round(cluster_remaining_ratio, 4),
                'forced_global': forced_global,
                'search_scope': scope_name,
                'scope_reason': scope_reason,
                'visited_count': self._visited_count(runtime_state),
                'local_loop_active': self._looping_on_local_subset(),
                'position_loop_active': self._position_loop_active(runtime_state),
                'cluster_revisit_streak': self._cluster_revisit_streak,
                'phase': runtime_state.phase,
                'known_civilians': known_civilians_count,
                'known_refuges': known_refuges_count,
                'active_detector_target': entity_value(runtime_state.detector_target),
                'search_mode': mode,
                'ml_low_confidence': low_confidence_ml,
                'ml_score_spread': None if ml_score_spread is None else round(ml_score_spread, 6),
                'ml_top_margin': None if ml_top_margin is None else round(ml_top_margin, 6),
                'previous_position': entity_value(runtime_state.previous_position),
                'loop_a': entity_value(loop_a),
                'loop_b': entity_value(loop_b),
            },
            candidate_payload,
            entity_value(chosen) if chosen is not None else None,
            chosen_reason,
            metadata=selection_metadata,
        )
        self._decision_logger.log_text(
            'Выбрана следующая зона поиска',
            {
                'цель': entity_value(chosen),
                'причина': chosen_reason,
                'scope': scope_name,
                'scope_reason': scope_reason,
                'forced_global': forced_global,
                'остаток_кластера': round(cluster_remaining_ratio, 4),
                'кандидатов': len(candidate_payload),
                'фаза': runtime_state.phase,
                'известных_жертв': known_civilians_count,
                'убежищ': known_refuges_count,
                'активная_цель_спасения': entity_value(runtime_state.detector_target),
                'предыдущая_позиция': entity_value(runtime_state.previous_position),
                'loop_a': entity_value(loop_a),
                'loop_b': entity_value(loop_b),
                'local_loop': self._looping_on_local_subset(),
                'position_loop': self._position_loop_active(runtime_state),
                'cluster_revisit_streak': self._cluster_revisit_streak,
                'топ': top_preview,
            },
        )
        self._decision_logger.state_snapshot('agent_runtime_state', runtime_state.snapshot())
        self._decision_logger.debug_event(
            'search_cycle_summary',
            {
                'compute_ms': compute_ms,
                'timings_ms': {key: round(value, 3) for key, value in timings.items()},
                'chosen': entity_value(chosen),
                'chosen_reason': chosen_reason,
                'scope_name': scope_name,
                'scope_reason': scope_reason,
                'candidate_count': len(candidate_payload),
                'stationary_ticks': runtime_state.stationary_ticks,
                'blocked_first_hops_from_current': runtime_state.blocked_first_hops_from(self._agent_info.get_position_entity_id(), tick),
                'current_target': entity_value(self._result),
                'previous_position': entity_value(runtime_state.previous_position),
                'loop_a': entity_value(loop_a),
                'loop_b': entity_value(loop_b),
                'local_loop_active': self._looping_on_local_subset(),
                'position_loop_active': self._position_loop_active(runtime_state),
                'cluster_revisit_streak': self._cluster_revisit_streak,
                'large_map': self._is_large_map,
                'selection_mode': selection_metadata.get('selection_mode'),
                'selected_by': selection_metadata.get('selected_by'),
                'exploration_used': selection_metadata.get('exploration_used'),
                'top_k_candidates': selection_metadata.get('top_k_candidates'),
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

    def _rank_candidates_for_logging(self, candidate_payload: list[dict[str, Any]]) -> None:
        for index, payload in enumerate(candidate_payload, start=1):
            payload['heuristic_rank'] = index
        ml_ranked = [item for item in candidate_payload if item.get('ml_score') is not None]
        ml_ranked.sort(key=lambda item: float(item.get('ml_score') or 0.0), reverse=True)
        for index, payload in enumerate(ml_ranked, start=1):
            payload['ml_rank'] = index
        for index, payload in enumerate(candidate_payload, start=1):
            payload['final_rank'] = index

    def _selection_metadata(self, candidate_payload: list[dict[str, Any]], chosen: Optional[EntityID], mode: str, selected_reason: str, *, scope_name: str = '', exploration_used: bool = False, selected_by: Optional[str] = None, diagnostics: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        shortlist_k = int(setting('search.ml_v2.shortlist_k', 8))
        top_k_candidates = [str(item.get('candidate_id')) for item in candidate_payload[:shortlist_k]]
        chosen_key = entity_value(chosen)
        selected_payload = next((item for item in candidate_payload if str(item.get('candidate_id')) == chosen_key), None)
        metadata = {
            'selection_mode': mode,
            'selected_by': selected_by or self._selection_source(mode, exploration_used),
            'exploration_used': exploration_used,
            'selected_reason': selected_reason,
            'search_scope': scope_name,
            'top_k_candidates': top_k_candidates,
            'selected_rank_by_heuristic': int(selected_payload.get('heuristic_rank', -1)) if selected_payload is not None else None,
            'selected_rank_by_ml': int(selected_payload.get('ml_rank', -1)) if selected_payload is not None and selected_payload.get('ml_rank') is not None else None,
            'selected_rank_by_final': int(selected_payload.get('final_rank', -1)) if selected_payload is not None else None,
        }
        if diagnostics:
            metadata.update(diagnostics)
        return metadata

    def get_target_entity_id(self) -> Optional[EntityID]:
        return self._result
