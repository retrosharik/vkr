from __future__ import annotations

import time

import pytest
from rcrscore.entities import EntityID, Road

from BaseRescueAgent.module.algorithm.logged_a_star_path_planning import LoggedAStarPathPlanning
from BaseRescueAgent.module.util.shared_runtime_state import AgentRuntimeState
from tests.conftest import FakeAgentInfo, FakeWorldInfo
from tests.unit.test_path_planning_logic import SilentLogger, StaticMl


def eid(value: int) -> EntityID:
    return EntityID(value)


def make_grid_world(size: int = 20):
    entities = []
    graph = {}
    for y in range(size):
        for x in range(size):
            node = y * size + x + 1
            neighbours = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    neighbours.append(ny * size + nx + 1)
            entities.append(Road(node, x=x * 1000, y=y * 1000, neighbours=neighbours))
            graph[eid(node)] = [eid(n) for n in neighbours]
    return FakeWorldInfo(entities), graph


@pytest.mark.load
def test_a_star_grid_micro_performance_under_threshold():
    world, graph = make_grid_world(size=20)
    planner = object.__new__(LoggedAStarPathPlanning)
    planner._world_info = world
    planner._agent_info = FakeAgentInfo(agent_id=1, position=1, tick=0)
    planner._graph = graph
    planner._decision_logger = SilentLogger()
    planner._ml = StaticMl()
    planner._cache = {}

    started = time.perf_counter()
    path, meta = planner._run_a_star(eid(1), eid(400), AgentRuntimeState(), tick=1, avoid_blocked_first_hops=False)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert path[0] == eid(1)
    assert path[-1] == eid(400)
    assert elapsed_ms < 100.0, f"A* on 20x20 grid took {elapsed_ms:.2f} ms"
    assert meta["expanded"] <= 400
