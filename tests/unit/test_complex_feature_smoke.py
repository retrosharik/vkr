from __future__ import annotations

import pytest
from rcrscore.entities import Building, Civilian, EntityID, Refuge, Road
from rcrscore.urn import EntityURN

from BaseRescueAgent.module.complex.priority_human_detector import PriorityHumanDetector
from BaseRescueAgent.module.complex.strategic_search import StrategicSearch
from BaseRescueAgent.module.util.shared_runtime_state import AgentRuntimeState
from tests.conftest import FakeAgentInfo, FakeHuman, FakeWorldInfo
from tests.unit.test_path_planning_logic import SilentLogger


def eid(value: int) -> EntityID:
    return EntityID(value)


@pytest.mark.unit
def test_detector_candidate_features_and_metadata_smoke():
    building = Building(10, x=0, y=0)
    refuge = Refuge(20, x=3000, y=0, urn=EntityURN.REFUGE)
    agent_body = FakeHuman(500, position=10, urn=EntityURN.AMBULANCE_TEAM)
    teammate = FakeHuman(501, position=10, urn=EntityURN.AMBULANCE_TEAM)
    civilian = Civilian(100, position=10, hp=9000, damage=30, buriedness=2)
    world = FakeWorldInfo([building, refuge, agent_body, teammate, civilian])
    agent = FakeAgentInfo(agent_id=500, position=10, tick=7)

    detector = object.__new__(PriorityHumanDetector)
    detector._world_info = world
    detector._agent_info = agent
    detector._result = eid(100)
    state = AgentRuntimeState()
    state.set_detector_target(eid(100), tick=5)

    features = detector._candidate_features(civilian, {eid(10)}, state, tick=7)
    metadata = detector._selection_metadata(
        [{**features, "heuristic_rank": 1, "ml_rank": 1, "final_rank": 1, "ml_score": 0.8, "final_score": 0.9}],
        eid(100),
        "hybrid",
        "unit",
    )

    assert features["candidate_id"] == "100"
    assert features["in_cluster"] is True
    assert features["current_target"] is True
    assert features["competitors"] == 1
    assert features["refuge_id"] == "20"
    assert metadata["selected_by"] == "hybrid"
    assert metadata["selected_ml_score"] == 0.8


@pytest.mark.unit
def test_search_candidate_features_and_metadata_smoke():
    building = Building(10, x=1000, y=0, neighbours=[1])
    road = Road(1, x=0, y=0, neighbours=[10])
    world = FakeWorldInfo([road, building])
    agent = FakeAgentInfo(agent_id=500, position=1, tick=1)

    search = object.__new__(StrategicSearch)
    search._world_info = world
    search._agent_info = agent
    search._is_large_map = False
    search._recent_targets = {eid(10)}
    search._result = eid(10)
    search._decision_logger = SilentLogger()
    search._distance_cached = lambda _from, _to: 1000.0
    search._centrality = lambda _building_id: 1
    search._hint_count = lambda _building_id, fast_only=False: 2
    search._known_problem_civilians = lambda _building_id: (1, 1)
    search._is_visited_building = lambda runtime_state, building_id: False

    state = AgentRuntimeState()
    features = search._candidate_features(building, {eid(10)}, state, "cluster", tick=1)
    payload = [{**features, "ml_score": 0.7}]
    search._rank_candidates_for_logging(payload)
    metadata = search._selection_metadata(payload, eid(10), "hybrid", "unit", scope_name="cluster", diagnostics={"diag": True})

    assert features["candidate_id"] == "10"
    assert features["civilian_hint_count"] == 2
    assert features["active_civilians_inside"] == 1
    assert features["recent_target"] is True
    assert payload[0]["heuristic_rank"] == 1
    assert payload[0]["ml_rank"] == 1
    assert metadata["selected_by"] == "hybrid"
    assert metadata["diag"] is True
