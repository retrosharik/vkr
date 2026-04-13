from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from rcrscore.entities import EntityID

from .decision_logger import entity_value


@dataclass
class AgentRuntimeState:
    phase: str = 'init'
    carrying: bool = False
    detector_target: Optional[EntityID] = None
    detector_target_since: int = -1
    search_target: Optional[EntityID] = None
    search_target_since: int = -1
    refuge_target: Optional[EntityID] = None
    refuge_target_since: int = -1
    blocked_search_targets: dict[str, int] = field(default_factory=dict)
    blocked_detector_targets: dict[str, int] = field(default_factory=dict)
    visited_buildings: set[str] = field(default_factory=set)
    entered_buildings: set[str] = field(default_factory=set)
    last_path_goal: Optional[EntityID] = None
    last_path_distance: Optional[float] = None
    last_path_tick: int = -1
    deferred_rescue_active: bool = False
    deferred_source_building: Optional[EntityID] = None
    deferred_target: Optional[EntityID] = None
    deferred_started_tick: int = -1
    deferred_deadline_tick: int = -1
    deferred_max_extra_buildings: int = 0
    deferred_swept_count: int = 0
    deferred_cooldown_until: int = -1
    deferred_reason: str = ''
    previous_position: Optional[EntityID] = None
    last_position: Optional[EntityID] = None
    startup_position: Optional[EntityID] = None
    startup_position_tick: int = -1
    first_real_move_tick: int = -1
    stationary_ticks: int = 0
    last_position_update_tick: int = -1
    last_moved_tick: int = -1
    recent_positions: deque[Optional[EntityID]] = field(default_factory=lambda: deque(maxlen=6))
    search_building_current: Optional[EntityID] = None
    search_building_entry_tick: int = -1
    search_building_last_hint_count: int = 0
    pending_move_start: Optional[EntityID] = None
    pending_move_goal: Optional[EntityID] = None
    pending_move_first_hop: Optional[EntityID] = None
    pending_move_tick: int = -1
    pending_move_context: str = ''
    pending_move_due_tick: int = -1
    pending_move_grace_ticks: int = 0
    pending_move_resolved: bool = True
    last_move_outcome: str = 'none'
    last_move_outcome_tick: int = -1
    last_failed_edge: str = ''
    failed_first_hops: dict[str, int] = field(default_factory=dict)
    blocked_first_hops: dict[str, int] = field(default_factory=dict)
    search_pending_selected: Optional[EntityID] = None
    search_pending_tick: int = -1
    search_pending_due_tick: int = -1
    search_pending_selected_rank: int = -1
    search_pending_ml_rank: int = -1
    search_pending_final_rank: int = -1
    search_pending_selected_by: str = 'heuristic'
    search_pending_selection_mode: str = 'heuristic'
    search_pending_reason: str = ''
    search_pending_scope: str = ''
    search_pending_exploration_used: bool = False
    search_pending_candidate_count: int = 0
    search_pending_top_k: list[str] = field(default_factory=list)
    search_pending_visited_count: int = 0
    search_outcome_queue: list[dict[str, Any]] = field(default_factory=list)
    detector_pending_selected: Optional[EntityID] = None
    detector_pending_tick: int = -1
    detector_pending_due_tick: int = -1
    detector_pending_selected_rank: int = -1
    detector_pending_ml_rank: int = -1
    detector_pending_final_rank: int = -1
    detector_pending_selected_by: str = 'heuristic'
    detector_pending_selection_mode: str = 'heuristic'
    detector_pending_reason: str = ''
    detector_pending_exploration_used: bool = False
    detector_pending_candidate_count: int = 0
    detector_pending_top_k: list[str] = field(default_factory=list)
    detector_pending_context: dict[str, Any] = field(default_factory=dict)
    detector_outcome_queue: list[dict[str, Any]] = field(default_factory=list)

    def cleanup(self, tick: int) -> None:
        self.blocked_search_targets = {key: value for key, value in self.blocked_search_targets.items() if value >= tick}
        self.blocked_detector_targets = {key: value for key, value in self.blocked_detector_targets.items() if value >= tick}
        self.blocked_first_hops = {key: value for key, value in self.blocked_first_hops.items() if value >= tick}

    def block_target(self, module_name: str, entity_id: Optional[EntityID], until_tick: int) -> None:
        key = entity_value(entity_id)
        if key == 'None':
            return
        if module_name == 'search':
            self.blocked_search_targets[key] = until_tick
        elif module_name == 'detector':
            self.blocked_detector_targets[key] = until_tick

    def is_blocked(self, module_name: str, entity_id: Optional[EntityID], tick: int) -> bool:
        key = entity_value(entity_id)
        if key == 'None':
            return False
        blocked_until = None
        if module_name == 'search':
            blocked_until = self.blocked_search_targets.get(key)
        elif module_name == 'detector':
            blocked_until = self.blocked_detector_targets.get(key)
        return blocked_until is not None and blocked_until >= tick

    def mark_visited_building(self, entity_id: Optional[EntityID]) -> bool:
        key = entity_value(entity_id)
        if key == 'None':
            return False
        before = len(self.visited_buildings)
        self.visited_buildings.add(key)
        return len(self.visited_buildings) > before

    def mark_entered_building(self, entity_id: Optional[EntityID]) -> bool:
        key = entity_value(entity_id)
        if key == 'None':
            return False
        before = len(self.entered_buildings)
        self.entered_buildings.add(key)
        return len(self.entered_buildings) > before

    def has_entered_building(self, entity_id: Optional[EntityID]) -> bool:
        key = entity_value(entity_id)
        if key == 'None':
            return False
        return key in self.entered_buildings

    def update_position(self, entity_id: Optional[EntityID], tick: Optional[int] = None) -> None:
        if tick is not None and self.last_position_update_tick == tick:
            return
        if tick is not None:
            self.last_position_update_tick = tick
        self.recent_positions.append(entity_id)
        if entity_id is None:
            self.previous_position = self.last_position
            self.last_position = None
            self.stationary_ticks = 0
            return
        if self.last_position is None:
            self.previous_position = None
            self.last_position = entity_id
            self.stationary_ticks = 0
            if self.startup_position is None:
                self.startup_position = entity_id
                if tick is not None:
                    self.startup_position_tick = tick
            return
        if self.last_position == entity_id:
            if tick is not None and self.pending_move_grace_active(tick):
                self.stationary_ticks = 0
                return
            self.stationary_ticks += 1
            return
        self.previous_position = self.last_position
        self.last_position = entity_id
        self.stationary_ticks = 0
        if self.startup_position is None:
            self.startup_position = self.previous_position
            if tick is not None:
                self.startup_position_tick = tick
        if self.first_real_move_tick < 0 and self.startup_position is not None and entity_id != self.startup_position:
            self.first_real_move_tick = tick if tick is not None else 0
            self.failed_first_hops.clear()
            self.blocked_first_hops.clear()
            self.pending_move_resolved = True
        if tick is not None:
            self.last_moved_tick = tick

    def has_real_movement(self) -> bool:
        return self.first_real_move_tick >= 0

    def startup_ticks_elapsed(self, tick: int) -> int:
        if self.startup_position_tick < 0:
            return 0
        return max(tick - self.startup_position_tick, 0)

    def startup_launch_active(self, tick: int, max_ticks: int) -> bool:
        return not self.has_real_movement() and self.startup_ticks_elapsed(tick) <= max(max_ticks, 0)

    def _edge_key(self, start: Optional[EntityID], first_hop: Optional[EntityID]) -> str:
        return f'{entity_value(start)}->{entity_value(first_hop)}'

    def block_first_hop(self, start: Optional[EntityID], first_hop: Optional[EntityID], until_tick: int) -> None:
        key = self._edge_key(start, first_hop)
        if key == 'None->None' or 'None' in key:
            return
        self.blocked_first_hops[key] = until_tick

    def is_first_hop_blocked(self, start: Optional[EntityID], first_hop: Optional[EntityID], tick: int) -> bool:
        key = self._edge_key(start, first_hop)
        blocked_until = self.blocked_first_hops.get(key)
        return blocked_until is not None and blocked_until >= tick

    def count_blocked_first_hops_from(self, start: Optional[EntityID], tick: int) -> int:
        prefix = f'{entity_value(start)}->'
        return sum(1 for key, until_tick in self.blocked_first_hops.items() if key.startswith(prefix) and until_tick >= tick)

    def blocked_first_hops_from(self, start: Optional[EntityID], tick: int) -> list[str]:
        prefix = f'{entity_value(start)}->'
        return sorted(key for key, until_tick in self.blocked_first_hops.items() if key.startswith(prefix) and until_tick >= tick)

    def pending_move_grace_active(self, tick: int) -> bool:
        return (
            self.pending_move_tick >= 0
            and not self.pending_move_resolved
            and self.pending_move_due_tick >= tick
        )

    def note_path_attempt(
        self,
        start: Optional[EntityID],
        goal: Optional[EntityID],
        path: list[EntityID],
        tick: int,
        context: str,
        grace_ticks: int = 0,
    ) -> None:
        first_hop = path[1] if len(path) >= 2 else None
        if (
            not self.pending_move_resolved
            and self.pending_move_start == start
            and self.pending_move_goal == goal
            and self.pending_move_first_hop == first_hop
            and self.pending_move_context == context
        ):
            return
        self.pending_move_start = start
        self.pending_move_goal = goal
        self.pending_move_first_hop = first_hop
        self.pending_move_tick = tick
        self.pending_move_context = context
        self.pending_move_grace_ticks = max(int(grace_ticks), 0)
        self.pending_move_due_tick = tick + self.pending_move_grace_ticks
        self.pending_move_resolved = False

    def resolve_pending_move_outcome(self, current_position: Optional[EntityID], tick: int, failure_threshold: int = 2, block_ticks: int = 8) -> Optional[dict[str, Any]]:
        if self.pending_move_tick < 0 or self.pending_move_resolved:
            return None
        start = self.pending_move_start
        first_hop = self.pending_move_first_hop
        edge_key = self._edge_key(start, first_hop)
        if start is not None and current_position is not None and current_position != start:
            self.last_move_outcome = 'moved'
            self.last_move_outcome_tick = tick
            self.last_failed_edge = ''
            if edge_key in self.failed_first_hops:
                self.failed_first_hops.pop(edge_key, None)
            self.pending_move_resolved = True
            return {
                'outcome': 'moved',
                'tick': tick,
                'from': entity_value(start),
                'to': entity_value(current_position),
                'goal': entity_value(self.pending_move_goal),
                'first_hop': entity_value(first_hop),
                'context': self.pending_move_context,
                'attempt_tick': self.pending_move_tick,
                'due_tick': self.pending_move_due_tick,
                'grace_ticks': self.pending_move_grace_ticks,
            }
        if self.pending_move_tick >= tick:
            return None
        if self.pending_move_due_tick >= tick:
            return None
        fail_count = self.failed_first_hops.get(edge_key, 0) + 1
        self.failed_first_hops[edge_key] = fail_count
        blocked_until = None
        if failure_threshold > 0 and fail_count >= failure_threshold and first_hop is not None and start is not None:
            blocked_until = tick + block_ticks
            self.block_first_hop(start, first_hop, blocked_until)
        self.last_move_outcome = 'stalled'
        self.last_move_outcome_tick = tick
        self.last_failed_edge = edge_key
        self.pending_move_resolved = True
        return {
            'outcome': 'stalled',
            'tick': tick,
            'from': entity_value(start),
            'goal': entity_value(self.pending_move_goal),
            'first_hop': entity_value(first_hop),
            'edge_key': edge_key,
            'context': self.pending_move_context,
            'fail_count': fail_count,
            'blocked_until': blocked_until,
            'attempt_tick': self.pending_move_tick,
            'due_tick': self.pending_move_due_tick,
            'grace_ticks': self.pending_move_grace_ticks,
        }

    def enter_search_building(self, entity_id: Optional[EntityID], tick: int, hint_count: int = 0) -> int:
        if entity_id is None:
            self.search_building_current = None
            self.search_building_entry_tick = -1
            self.search_building_last_hint_count = 0
            return 0
        self.mark_entered_building(entity_id)
        if self.search_building_current != entity_id:
            self.search_building_current = entity_id
            self.search_building_entry_tick = tick
            self.search_building_last_hint_count = hint_count
            return 0
        if hint_count > self.search_building_last_hint_count:
            self.search_building_last_hint_count = hint_count
        return max(tick - self.search_building_entry_tick, 0)

    def leave_search_building(self) -> None:
        self.search_building_current = None
        self.search_building_entry_tick = -1
        self.search_building_last_hint_count = 0

    def search_building_dwell_ticks(self, entity_id: Optional[EntityID], tick: int) -> int:
        if entity_id is None or self.search_building_current != entity_id or self.search_building_entry_tick < 0:
            return 0
        return max(tick - self.search_building_entry_tick, 0)

    def set_search_target(self, entity_id: Optional[EntityID], tick: int) -> None:
        if self.search_target == entity_id:
            if entity_id is None:
                self.search_target_since = -1
            return
        self.search_target = entity_id
        self.search_target_since = tick if entity_id is not None else -1

    def set_detector_target(self, entity_id: Optional[EntityID], tick: int) -> None:
        if self.detector_target == entity_id:
            if entity_id is None:
                self.detector_target_since = -1
            return
        self.detector_target = entity_id
        self.detector_target_since = tick if entity_id is not None else -1

    def set_refuge_target(self, entity_id: Optional[EntityID], tick: int) -> None:
        if self.refuge_target == entity_id:
            if entity_id is None:
                self.refuge_target_since = -1
            return
        self.refuge_target = entity_id
        self.refuge_target_since = tick if entity_id is not None else -1

    def _build_search_outcome_result(self, tick: int, *, due_reached: bool, target_visited: bool, detector_target_found: bool, resolved_reason: str) -> dict[str, Any]:
        return {
            "selected_id": entity_value(self.search_pending_selected),
            "selection_tick": self.search_pending_tick,
            "resolved_tick": tick,
            "elapsed_ticks": max(tick - self.search_pending_tick, 0),
            "visited_count_gain": len(self.entered_buildings) - self.search_pending_visited_count,
            "target_visited": target_visited,
            "detector_target_found": detector_target_found,
            "phase": self.phase,
            "selected_rank_by_heuristic": self.search_pending_selected_rank,
            "selected_rank_by_ml": self.search_pending_ml_rank,
            "selected_rank_by_final": self.search_pending_final_rank,
            "selected_by": self.search_pending_selected_by,
            "selection_mode": self.search_pending_selection_mode,
            "selected_reason": self.search_pending_reason,
            "search_scope": self.search_pending_scope,
            "exploration_used": self.search_pending_exploration_used,
            "candidate_count": self.search_pending_candidate_count,
            "top_k_candidates": list(self.search_pending_top_k),
            "due_reached": due_reached,
            "resolved_reason": resolved_reason,
        }

    def _clear_search_pending(self) -> None:
        self.search_pending_selected = None
        self.search_pending_tick = -1
        self.search_pending_due_tick = -1
        self.search_pending_selected_rank = -1
        self.search_pending_ml_rank = -1
        self.search_pending_final_rank = -1
        self.search_pending_selected_by = 'heuristic'
        self.search_pending_selection_mode = 'heuristic'
        self.search_pending_reason = ''
        self.search_pending_scope = ''
        self.search_pending_exploration_used = False
        self.search_pending_candidate_count = 0
        self.search_pending_top_k = []
        self.search_pending_visited_count = len(self.entered_buildings)

    def _build_detector_outcome_result(self, tick: int, *, due_reached: bool, carrying_now: bool, target_changed: bool, refuge_target_set: bool, resolved_reason: str) -> dict[str, Any]:
        selected_key = entity_value(self.detector_pending_selected)
        current_key = entity_value(self.detector_target)
        result = {
            "selected_id": selected_key,
            "selection_tick": self.detector_pending_tick,
            "resolved_tick": tick,
            "elapsed_ticks": max(tick - self.detector_pending_tick, 0),
            "carrying_now": carrying_now,
            "refuge_target_set": refuge_target_set,
            "phase": self.phase,
            "target_still_active": selected_key == current_key,
            "target_changed": target_changed,
            "selected_rank_by_heuristic": self.detector_pending_selected_rank,
            "selected_rank_by_ml": self.detector_pending_ml_rank,
            "selected_rank_by_final": self.detector_pending_final_rank,
            "selected_by": self.detector_pending_selected_by,
            "selection_mode": self.detector_pending_selection_mode,
            "selected_reason": self.detector_pending_reason,
            "exploration_used": self.detector_pending_exploration_used,
            "candidate_count": self.detector_pending_candidate_count,
            "top_k_candidates": list(self.detector_pending_top_k),
            "due_reached": due_reached,
            "resolved_reason": resolved_reason,
        }
        for key, value in self.detector_pending_context.items():
            result[f'selection_{key}'] = value
        return result

    def _clear_detector_pending(self) -> None:
        self.detector_pending_selected = None
        self.detector_pending_tick = -1
        self.detector_pending_due_tick = -1
        self.detector_pending_selected_rank = -1
        self.detector_pending_ml_rank = -1
        self.detector_pending_final_rank = -1
        self.detector_pending_selected_by = 'heuristic'
        self.detector_pending_selection_mode = 'heuristic'
        self.detector_pending_reason = ''
        self.detector_pending_exploration_used = False
        self.detector_pending_candidate_count = 0
        self.detector_pending_top_k = []
        self.detector_pending_context = {}

    def register_search_selection(
        self,
        selected: Optional[EntityID],
        tick: int,
        outcome_window_ticks: int,
        *,
        selected_rank: int = -1,
        ml_rank: int = -1,
        final_rank: int = -1,
        selected_by: str = 'heuristic',
        selection_mode: str = 'heuristic',
        reason: str = '',
        scope: str = '',
        exploration_used: bool = False,
        candidate_count: int = 0,
        top_k_candidates: list[str] | None = None,
    ) -> None:
        pending_key = entity_value(self.search_pending_selected)
        selected_key = entity_value(selected)
        if self.search_pending_tick >= 0 and selected_key == pending_key and selected_key != 'None':
            return
        if self.search_pending_tick >= 0 and tick > self.search_pending_tick:
            target_visited = pending_key != 'None' and self.has_entered_building(self.search_pending_selected)
            detector_target_found = self.detector_target is not None
            self.search_outcome_queue.append(
                self._build_search_outcome_result(
                    tick,
                    due_reached=False,
                    target_visited=target_visited,
                    detector_target_found=detector_target_found,
                    resolved_reason='superseded',
                )
            )
            self._clear_search_pending()
        self.search_pending_selected = selected
        self.search_pending_tick = tick
        self.search_pending_due_tick = tick + max(int(outcome_window_ticks), 1)
        self.search_pending_selected_rank = int(selected_rank)
        self.search_pending_ml_rank = int(ml_rank)
        self.search_pending_final_rank = int(final_rank)
        self.search_pending_selected_by = str(selected_by)
        self.search_pending_selection_mode = str(selection_mode)
        self.search_pending_reason = str(reason)
        self.search_pending_scope = str(scope)
        self.search_pending_exploration_used = bool(exploration_used)
        self.search_pending_candidate_count = int(candidate_count)
        self.search_pending_top_k = list(top_k_candidates or [])
        self.search_pending_visited_count = len(self.entered_buildings)

    def register_detector_selection(
        self,
        selected: Optional[EntityID],
        tick: int,
        outcome_window_ticks: int,
        *,
        selected_rank: int = -1,
        ml_rank: int = -1,
        final_rank: int = -1,
        selected_by: str = 'heuristic',
        selection_mode: str = 'heuristic',
        reason: str = '',
        exploration_used: bool = False,
        candidate_count: int = 0,
        top_k_candidates: list[str] | None = None,
        selection_context: dict[str, Any] | None = None,
    ) -> None:
        pending_key = entity_value(self.detector_pending_selected)
        selected_key = entity_value(selected)
        if self.detector_pending_tick >= 0 and selected_key == pending_key and selected_key != 'None':
            return
        if self.detector_pending_tick >= 0 and tick > self.detector_pending_tick:
            carrying_now = bool(self.carrying)
            refuge_target_set = self.refuge_target is not None
            target_changed = pending_key != entity_value(self.detector_target)
            self.detector_outcome_queue.append(
                self._build_detector_outcome_result(
                    tick,
                    due_reached=False,
                    carrying_now=carrying_now,
                    target_changed=target_changed,
                    refuge_target_set=refuge_target_set,
                    resolved_reason='superseded',
                )
            )
            self._clear_detector_pending()
        self.detector_pending_selected = selected
        self.detector_pending_tick = tick
        self.detector_pending_due_tick = tick + max(int(outcome_window_ticks), 1)
        self.detector_pending_selected_rank = int(selected_rank)
        self.detector_pending_ml_rank = int(ml_rank)
        self.detector_pending_final_rank = int(final_rank)
        self.detector_pending_selected_by = str(selected_by)
        self.detector_pending_selection_mode = str(selection_mode)
        self.detector_pending_reason = str(reason)
        self.detector_pending_exploration_used = bool(exploration_used)
        self.detector_pending_candidate_count = int(candidate_count)
        self.detector_pending_top_k = list(top_k_candidates or [])
        self.detector_pending_context = dict(selection_context or {})

    def collect_search_outcomes(self, tick: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if self.search_pending_tick >= 0 and tick > self.search_pending_tick:
            selected_key = entity_value(self.search_pending_selected)
            target_visited = selected_key != 'None' and self.has_entered_building(self.search_pending_selected)
            due_reached = tick >= self.search_pending_due_tick
            detector_target_found = self.detector_target is not None
            if due_reached or target_visited or detector_target_found:
                results.append(
                    self._build_search_outcome_result(
                        tick,
                        due_reached=due_reached,
                        target_visited=target_visited,
                        detector_target_found=detector_target_found,
                        resolved_reason='window_or_progress',
                    )
                )
                self._clear_search_pending()
        if self.search_outcome_queue:
            results = list(self.search_outcome_queue) + results
            self.search_outcome_queue = []
        return results

    def maybe_resolve_search_outcome(self, tick: int) -> Optional[dict[str, Any]]:
        results = self.collect_search_outcomes(tick)
        return results[0] if results else None

    def collect_detector_outcomes(self, tick: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if self.detector_pending_tick >= 0 and tick > self.detector_pending_tick:
            selected_key = entity_value(self.detector_pending_selected)
            current_key = entity_value(self.detector_target)
            due_reached = tick >= self.detector_pending_due_tick
            carrying_now = bool(self.carrying)
            target_changed = selected_key != current_key
            refuge_target_set = self.refuge_target is not None
            if due_reached or carrying_now or target_changed or refuge_target_set:
                results.append(
                    self._build_detector_outcome_result(
                        tick,
                        due_reached=due_reached,
                        carrying_now=carrying_now,
                        target_changed=target_changed,
                        refuge_target_set=refuge_target_set,
                        resolved_reason='window_or_progress',
                    )
                )
                self._clear_detector_pending()
        if self.detector_outcome_queue:
            results = list(self.detector_outcome_queue) + results
            self.detector_outcome_queue = []
        return results

    def maybe_resolve_detector_outcome(self, tick: int) -> Optional[dict[str, Any]]:
        results = self.collect_detector_outcomes(tick)
        return results[0] if results else None

    def snapshot(self) -> dict[str, Any]:
        return {
            'phase': self.phase,
            'carrying': self.carrying,
            'search_target': entity_value(self.search_target),
            'search_target_since': self.search_target_since,
            'detector_target': entity_value(self.detector_target),
            'detector_target_since': self.detector_target_since,
            'refuge_target': entity_value(self.refuge_target),
            'refuge_target_since': self.refuge_target_since,
            'blocked_search_targets': dict(self.blocked_search_targets),
            'blocked_detector_targets': dict(self.blocked_detector_targets),
            'visited_buildings': sorted(self.visited_buildings),
            'entered_buildings': sorted(self.entered_buildings),
            'last_path_goal': entity_value(self.last_path_goal),
            'last_path_distance': self.last_path_distance,
            'last_path_tick': self.last_path_tick,
            'deferred_rescue_active': self.deferred_rescue_active,
            'deferred_source_building': entity_value(self.deferred_source_building),
            'deferred_target': entity_value(self.deferred_target),
            'deferred_started_tick': self.deferred_started_tick,
            'deferred_deadline_tick': self.deferred_deadline_tick,
            'deferred_max_extra_buildings': self.deferred_max_extra_buildings,
            'deferred_swept_count': self.deferred_swept_count,
            'deferred_cooldown_until': self.deferred_cooldown_until,
            'deferred_reason': self.deferred_reason,
            'previous_position': entity_value(self.previous_position),
            'last_position': entity_value(self.last_position),
            'startup_position': entity_value(self.startup_position),
            'startup_position_tick': self.startup_position_tick,
            'first_real_move_tick': self.first_real_move_tick,
            'has_real_movement': self.has_real_movement(),
            'stationary_ticks': self.stationary_ticks,
            'last_position_update_tick': self.last_position_update_tick,
            'last_moved_tick': self.last_moved_tick,
            'recent_positions': [entity_value(p) for p in self.recent_positions],
            'search_building_current': entity_value(self.search_building_current),
            'search_building_entry_tick': self.search_building_entry_tick,
            'search_building_last_hint_count': self.search_building_last_hint_count,
            'pending_move_start': entity_value(self.pending_move_start),
            'pending_move_goal': entity_value(self.pending_move_goal),
            'pending_move_first_hop': entity_value(self.pending_move_first_hop),
            'pending_move_tick': self.pending_move_tick,
            'pending_move_context': self.pending_move_context,
            'pending_move_resolved': self.pending_move_resolved,
            'last_move_outcome': self.last_move_outcome,
            'last_move_outcome_tick': self.last_move_outcome_tick,
            'last_failed_edge': self.last_failed_edge,
            'failed_first_hops': dict(self.failed_first_hops),
            'blocked_first_hops': dict(self.blocked_first_hops),
            'search_pending_selected': entity_value(self.search_pending_selected),
            'search_pending_tick': self.search_pending_tick,
            'search_pending_due_tick': self.search_pending_due_tick,
            'search_pending_selected_rank': self.search_pending_selected_rank,
            'search_pending_ml_rank': self.search_pending_ml_rank,
            'search_pending_final_rank': self.search_pending_final_rank,
            'search_pending_selected_by': self.search_pending_selected_by,
            'search_pending_selection_mode': self.search_pending_selection_mode,
            'search_pending_reason': self.search_pending_reason,
            'search_pending_scope': self.search_pending_scope,
            'search_pending_exploration_used': self.search_pending_exploration_used,
            'search_pending_candidate_count': self.search_pending_candidate_count,
            'search_pending_top_k': list(self.search_pending_top_k),
            'search_outcome_queue_size': len(self.search_outcome_queue),
            'search_pending_visited_count': self.search_pending_visited_count,
            'detector_pending_selected': entity_value(self.detector_pending_selected),
            'detector_pending_tick': self.detector_pending_tick,
            'detector_pending_due_tick': self.detector_pending_due_tick,
            'detector_pending_selected_rank': self.detector_pending_selected_rank,
            'detector_pending_ml_rank': self.detector_pending_ml_rank,
            'detector_pending_final_rank': self.detector_pending_final_rank,
            'detector_pending_selected_by': self.detector_pending_selected_by,
            'detector_pending_selection_mode': self.detector_pending_selection_mode,
            'detector_pending_reason': self.detector_pending_reason,
            'detector_pending_exploration_used': self.detector_pending_exploration_used,
            'detector_pending_candidate_count': self.detector_pending_candidate_count,
            'detector_pending_top_k': list(self.detector_pending_top_k),
            'detector_outcome_queue_size': len(self.detector_outcome_queue),
        }


_STATE_BY_AGENT: dict[str, AgentRuntimeState] = {}


def get_runtime_state(agent_info) -> AgentRuntimeState:
    getter = getattr(agent_info, 'get_entity_id', None)
    key = entity_value(getter() if callable(getter) else None)
    if key not in _STATE_BY_AGENT:
        _STATE_BY_AGENT[key] = AgentRuntimeState()
    return _STATE_BY_AGENT[key]
