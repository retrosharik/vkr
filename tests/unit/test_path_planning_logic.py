from __future__ import annotations

import pytest
from rcrscore.entities import Building, EntityID, Road

from BaseRescueAgent.module.algorithm.logged_a_star_path_planning import LoggedAStarPathPlanning
from BaseRescueAgent.module.util.shared_runtime_state import AgentRuntimeState
from tests.conftest import FakeAgentInfo, FakeWorldInfo


def eid(value: int) -> EntityID:
    return EntityID(value)


class SilentLogger:
    def debug_event(self, *args, **kwargs):
        pass

    def path_snapshot(self, *args, **kwargs):
        pass

    def log_text(self, *args, **kwargs):
        pass

    def debug_text(self, *args, **kwargs):
        pass


class StaticMl:
    def mode_name(self):
        return "heuristic"

    def describe(self):
        return {}


def make_planner(world_info, graph):
    planner = object.__new__(LoggedAStarPathPlanning)
    planner._world_info = world_info
    planner._agent_info = FakeAgentInfo(agent_id=1, position=1, tick=0)
    planner._graph = graph
    planner._decision_logger = SilentLogger()
    planner._ml = StaticMl()
    planner._cache = {}
    return planner


@pytest.mark.unit
def test_run_a_star_finds_shortest_route_on_small_graph():
    r1 = Road(1, x=0, y=0, neighbours=[2, 4])
    r2 = Road(2, x=1000, y=0, neighbours=[1, 3])
    r3 = Building(3, x=2000, y=0, neighbours=[2, 4])
    r4 = Road(4, x=0, y=10000, neighbours=[1, 3])
    world = FakeWorldInfo([r1, r2, r3, r4])
    graph = {eid(1): [eid(2), eid(4)], eid(2): [eid(1), eid(3)], eid(3): [eid(2), eid(4)], eid(4): [eid(1), eid(3)]}
    planner = make_planner(world, graph)

    path, meta = planner._run_a_star(eid(1), eid(3), AgentRuntimeState(), tick=1, avoid_blocked_first_hops=False)

    assert path == [eid(1), eid(2), eid(3)]
    assert meta["expanded"] >= 2
    assert meta["total_cost"] is not None


@pytest.mark.unit
def test_run_a_star_avoids_blocked_first_hop_when_enabled():
    r1 = Road(1, x=0, y=0, neighbours=[2, 4])
    r2 = Road(2, x=1000, y=0, neighbours=[1, 3])
    r3 = Building(3, x=2000, y=0, neighbours=[2, 4])
    r4 = Road(4, x=0, y=1000, neighbours=[1, 3])
    world = FakeWorldInfo([r1, r2, r3, r4])
    graph = {eid(1): [eid(2), eid(4)], eid(2): [eid(1), eid(3)], eid(3): [eid(2), eid(4)], eid(4): [eid(1), eid(3)]}
    state = AgentRuntimeState()
    state.block_first_hop(eid(1), eid(2), until_tick=10)
    planner = make_planner(world, graph)

    path, meta = planner._run_a_star(eid(1), eid(3), state, tick=5, avoid_blocked_first_hops=True)

    assert path == [eid(1), eid(4), eid(3)]
    assert "1->2" in meta["skipped_start_edges"]


@pytest.mark.unit
def test_candidate_final_cost_penalizes_failed_or_blocked_first_hop():
    world = FakeWorldInfo([Road(1), Road(2)])
    planner = make_planner(world, {eid(1): [eid(2)]})
    state = AgentRuntimeState()
    state.failed_first_hops["1->2"] = 2
    state.block_first_hop(eid(1), eid(2), until_tick=10)

    cost = planner._candidate_final_cost(base_cost=100.0, risk=0.5, runtime_state=state, start=eid(1), first_hop=eid(2))

    assert cost > 100.0


@pytest.mark.unit
def test_should_not_override_when_first_hop_is_same():
    world = FakeWorldInfo([Road(1), Road(2)])
    planner = make_planner(world, {eid(1): [eid(2)]})

    allowed, reason, debug = planner._should_override_first_hop(
        {"first_hop": eid(2), "risk": 0.5, "base_cost": 100.0, "final_cost": 100.0},
        {"first_hop": eid(2), "risk": 0.1, "base_cost": 90.0, "final_cost": 90.0},
        caller_context="search",
    )

    assert allowed is False
    assert reason == "same_first_hop"
    assert debug["would_override_reason_if_enabled"] == "same_first_hop"
