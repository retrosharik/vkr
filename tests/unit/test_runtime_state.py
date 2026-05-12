from __future__ import annotations

import pytest
from rcrscore.entities import EntityID

from BaseRescueAgent.module.util.shared_runtime_state import AgentRuntimeState, get_runtime_state, _STATE_BY_AGENT


def eid(value: int) -> EntityID:
    return EntityID(value)


@pytest.mark.unit
def test_blocked_targets_expire_after_cleanup():
    state = AgentRuntimeState()
    target = eid(101)

    state.block_target("search", target, until_tick=10)

    assert state.is_blocked("search", target, tick=10)
    assert not state.is_blocked("search", target, tick=11)

    state.cleanup(tick=11)
    assert state.blocked_search_targets == {}


@pytest.mark.unit
def test_position_tracking_detects_first_real_move_and_stationary_ticks():
    state = AgentRuntimeState()

    state.update_position(eid(1), tick=1)
    state.update_position(eid(1), tick=2)
    state.update_position(eid(1), tick=3)

    assert state.startup_position == eid(1)
    assert state.first_real_move_tick == -1
    assert state.stationary_ticks == 2

    state.update_position(eid(2), tick=4)

    assert state.has_real_movement()
    assert state.first_real_move_tick == 4
    assert state.previous_position == eid(1)
    assert state.last_position == eid(2)
    assert state.stationary_ticks == 0


@pytest.mark.unit
def test_position_update_is_idempotent_within_same_tick():
    state = AgentRuntimeState()

    state.update_position(eid(1), tick=5)
    state.update_position(eid(1), tick=5)

    assert state.stationary_ticks == 0
    assert list(state.recent_positions) == [eid(1)]


@pytest.mark.unit
def test_failed_first_hop_is_blocked_after_threshold():
    state = AgentRuntimeState()
    start = eid(1)
    first_hop = eid(2)
    goal = eid(3)
    path = [start, first_hop, goal]

    state.note_path_attempt(start, goal, path, tick=10, context="action_move", grace_ticks=0)
    outcome1 = state.resolve_pending_move_outcome(start, tick=11, failure_threshold=2, block_ticks=5)

    state.note_path_attempt(start, goal, path, tick=12, context="action_move", grace_ticks=0)
    outcome2 = state.resolve_pending_move_outcome(start, tick=13, failure_threshold=2, block_ticks=5)

    assert outcome1["outcome"] == "stalled"
    assert outcome1["blocked_until"] is None
    assert outcome2["outcome"] == "stalled"
    assert outcome2["blocked_until"] == 18
    assert state.is_first_hop_blocked(start, first_hop, tick=18)
    assert not state.is_first_hop_blocked(start, first_hop, tick=19)


@pytest.mark.unit
def test_successful_movement_resolves_pending_attempt_and_clears_failure_count():
    state = AgentRuntimeState()
    start = eid(1)
    first_hop = eid(2)
    goal = eid(3)
    edge_key = "1->2"
    state.failed_first_hops[edge_key] = 1

    state.note_path_attempt(start, goal, [start, first_hop, goal], tick=10, context="action_move", grace_ticks=0)
    outcome = state.resolve_pending_move_outcome(first_hop, tick=11, failure_threshold=2, block_ticks=5)

    assert outcome["outcome"] == "moved"
    assert state.last_move_outcome == "moved"
    assert edge_key not in state.failed_first_hops
    assert state.pending_move_resolved is True


@pytest.mark.unit
def test_search_outcome_is_emitted_when_target_entered():
    state = AgentRuntimeState()
    target = eid(100)

    state.register_search_selection(
        target,
        tick=20,
        outcome_window_ticks=10,
        selected_rank=1,
        ml_rank=2,
        final_rank=1,
        selected_by="hybrid",
        selection_mode="safe_rerank",
        reason="unit-test",
        scope="cluster",
        candidate_count=3,
        top_k_candidates=["100", "101"],
    )
    state.mark_entered_building(target)
    outcomes = state.collect_search_outcomes(tick=21)

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome["selected_id"] == "100"
    assert outcome["target_visited"] is True
    assert outcome["visited_count_gain"] == 1
    assert outcome["selected_rank_by_final"] == 1
    assert outcome["resolved_reason"] == "window_or_progress"


@pytest.mark.unit
def test_detector_outcome_is_emitted_when_agent_starts_carrying():
    state = AgentRuntimeState()
    target = eid(200)
    state.set_detector_target(target, tick=1)

    state.register_detector_selection(
        target,
        tick=30,
        outcome_window_ticks=8,
        selected_rank=1,
        ml_rank=1,
        final_rank=1,
        selected_by="ml_override",
        selection_mode="hybrid",
        reason="unit-test",
        candidate_count=2,
        top_k_candidates=["200", "201"],
        selection_context={"life_margin": 42.0},
    )
    state.carrying = True
    outcomes = state.collect_detector_outcomes(tick=31)

    assert len(outcomes) == 1
    assert outcomes[0]["selected_id"] == "200"
    assert outcomes[0]["carrying_now"] is True
    assert outcomes[0]["selection_life_margin"] == 42.0


@pytest.mark.unit
def test_runtime_state_is_scoped_by_agent_id():
    class AgentInfo:
        def __init__(self, value):
            self.value = eid(value)

        def get_entity_id(self):
            return self.value

    _STATE_BY_AGENT.clear()
    state_a = get_runtime_state(AgentInfo(1))
    state_b = get_runtime_state(AgentInfo(2))
    state_a.phase = "search"
    state_b.phase = "transport"

    assert get_runtime_state(AgentInfo(1)).phase == "search"
    assert get_runtime_state(AgentInfo(2)).phase == "transport"
