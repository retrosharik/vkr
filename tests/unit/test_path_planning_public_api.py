from __future__ import annotations

from collections import OrderedDict

import pytest
from rcrscore.entities import Building, EntityID, Road

from BaseRescueAgent.module.algorithm.logged_a_star_path_planning import LoggedAStarPathPlanning
from BaseRescueAgent.module.util.shared_runtime_state import _STATE_BY_AGENT
from tests.conftest import FakeAgentInfo, FakeWorldInfo
from tests.unit.test_path_planning_logic import SilentLogger, StaticMl


def eid(value: int) -> EntityID:
    return EntityID(value)


def make_planner(world_info, graph, *, position=1, tick=1):
    planner = object.__new__(LoggedAStarPathPlanning)
    planner._world_info = world_info
    planner._agent_info = FakeAgentInfo(agent_id=99, position=position, tick=tick)
    planner._graph = graph
    planner._decision_logger = SilentLogger()
    planner._ml = StaticMl()
    planner._cache = OrderedDict()
    return planner


@pytest.fixture(autouse=True)
def clear_runtime_state():
    _STATE_BY_AGENT.clear()
    yield
    _STATE_BY_AGENT.clear()


@pytest.mark.unit
def test_get_path_handles_invalid_and_trivial_requests():
    world = FakeWorldInfo([Road(1)])
    planner = make_planner(world, {eid(1): []})

    assert planner.get_path(eid(999), eid(1)) == []
    assert planner.get_path(eid(1), eid(1)) == [eid(1)]


@pytest.mark.unit
def test_get_path_uses_cache_for_repeated_request():
    r1 = Road(1, x=0, y=0, neighbours=[2])
    r2 = Road(2, x=1000, y=0, neighbours=[1, 3])
    b3 = Building(3, x=2000, y=0, neighbours=[2])
    world = FakeWorldInfo([r1, r2, b3])
    graph = {eid(1): [eid(2)], eid(2): [eid(1), eid(3)], eid(3): [eid(2)]}
    planner = make_planner(world, graph)

    first = planner.get_path(eid(1), eid(3))
    second = planner.get_path(eid(1), eid(3))

    assert first == [eid(1), eid(2), eid(3)]
    assert second == first
    assert len(planner._cache) == 1


@pytest.mark.unit
def test_get_path_to_multiple_destinations_selects_nearest_reachable_goal():
    r1 = Road(1, x=0, y=0, neighbours=[2, 4])
    r2 = Road(2, x=1000, y=0, neighbours=[1, 3])
    b3 = Building(3, x=2000, y=0, neighbours=[2])
    r4 = Road(4, x=0, y=1000, neighbours=[1, 5])
    b5 = Building(5, x=0, y=2000, neighbours=[4])
    world = FakeWorldInfo([r1, r2, b3, r4, b5])
    graph = {
        eid(1): [eid(2), eid(4)],
        eid(2): [eid(1), eid(3)],
        eid(3): [eid(2)],
        eid(4): [eid(1), eid(5)],
        eid(5): [eid(4)],
    }
    planner = make_planner(world, graph)

    assert planner.get_path_to_multiple_destinations(eid(1), {eid(3), eid(5)}) in (
        [eid(1), eid(2), eid(3)],
        [eid(1), eid(4), eid(5)],
    )
    assert planner.get_distance(eid(1), eid(3)) == pytest.approx(2000.0)
