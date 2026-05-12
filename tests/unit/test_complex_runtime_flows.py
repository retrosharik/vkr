from __future__ import annotations

from collections import deque, OrderedDict

import pytest
from rcrscore.entities import Building, Civilian, EntityID, Refuge, Road
from rcrscore.urn import EntityURN

from BaseRescueAgent.module.algorithm.logged_a_star_path_planning import LoggedAStarPathPlanning
from BaseRescueAgent.module.algorithm.ml_path_planning import MlPathPlanning
from BaseRescueAgent.module.complex.ml_detector import MlDetector
from BaseRescueAgent.module.complex.ml_search import MlSearch
from BaseRescueAgent.module.complex.priority_human_detector import PriorityHumanDetector
from BaseRescueAgent.module.complex.strategic_search import StrategicSearch
from BaseRescueAgent.module.util.shared_runtime_state import AgentRuntimeState, _STATE_BY_AGENT, get_runtime_state
from tests.conftest import FakeAgentInfo, FakeHuman, FakeWorldInfo
from tests.unit.test_path_planning_logic import SilentLogger, StaticMl


def eid(value: int) -> EntityID:
    return EntityID(value)


class CaptureLogger(SilentLogger):
    def __init__(self):
        self.events = []
        self.snapshots = []
        self.text = []

    def debug_event(self, event_type, payload=None):
        self.events.append((event_type, payload or {}))

    def decision_snapshot(self, *args, **kwargs):
        self.snapshots.append((args, kwargs))

    def state_snapshot(self, *args, **kwargs):
        self.snapshots.append((args, kwargs))

    def log_text(self, message, payload=None):
        self.text.append((message, payload or {}))

    def path_snapshot(self, *args, **kwargs):
        self.snapshots.append((args, kwargs))


class FakeClustering:
    def __init__(self, entities):
        self.entities = list(entities)

    def get_cluster_index(self, agent_id):
        return 0

    def get_cluster_entities(self, index):
        return list(self.entities)


class FakePathPlanning:
    def __init__(self, reachable=True):
        self.reachable = reachable
        self.calls = []

    def get_path(self, start, goal):
        self.calls.append((start, goal))
        if not self.reachable:
            return []
        if start == goal:
            return [start]
        return [start, goal]


class HybridMl:
    requested_mode = "hybrid"

    def __init__(self, scores=None):
        self.scores = scores or {}

    def mode_name(self):
        return "hybrid"

    def describe(self):
        return {"active": True, "mode": "hybrid"}

    def score_candidates(self, context, candidates):
        return {str(item.get("candidate_id")): self.scores.get(str(item.get("candidate_id")), 0.5) for item in candidates}

    def score_path(self, context, payload):
        return 0.2


class HeuristicMl:
    requested_mode = "heuristic"

    def mode_name(self):
        return "heuristic"

    def describe(self):
        return {"active": False, "mode": "heuristic"}

    def score_candidates(self, context, candidates):
        return {}

    def score_path(self, context, payload):
        return None

@pytest.fixture(autouse=True)
def clear_state():
    _STATE_BY_AGENT.clear()
    yield
    _STATE_BY_AGENT.clear()


def make_detector(world, agent, *, cluster_entities=(), ml=None, reachable=True):
    detector = object.__new__(PriorityHumanDetector)
    detector._world_info = world
    detector._agent_info = agent
    detector._clustering = FakeClustering(cluster_entities)
    detector._path_planning = FakePathPlanning(reachable=reachable)
    detector._decision_logger = CaptureLogger()
    detector._ml = ml or HeuristicMl()
    detector._result = None
    return detector


def make_search(world, agent, *, buildings, cluster_buildings=None, ml=None, reachable=True):
    search = object.__new__(StrategicSearch)
    search._world_info = world
    search._agent_info = agent
    search._clustering = FakeClustering(cluster_buildings or buildings)
    search._path_planning = FakePathPlanning(reachable=reachable)
    search._decision_logger = CaptureLogger()
    search._ml = ml or HeuristicMl()
    search._visited = set()
    search._recent_targets = deque(maxlen=8)
    search._recent_progress = deque(maxlen=6)
    search._result = None
    search._static_all_buildings = list(buildings)
    search._static_cluster_buildings = list(cluster_buildings or buildings)
    search._static_cluster_ids = {b.get_entity_id() for b in (cluster_buildings or buildings)}
    search._coord_cache = {str(b.get_entity_id().get_value()): (b.get_x(), b.get_y()) for b in buildings}
    search._centrality_cache = {}
    search._tick_cache_tick = -1
    search._distance_cache = {}
    search._approx_distance_cache = {}
    search._hint_cache_exact = {}
    search._hint_cache_fast = {}
    search._visible_civilians_cache = []
    search._visible_civilian_positions = {}
    search._all_buildings_count = len(buildings)
    search._is_large_map = False
    search._agent_xy = (None, None)
    search._last_visited_count = 0
    search._last_area_key = None
    search._forced_global_until = -1
    search._cluster_revisit_streak = 0
    search._ml_status_logged = False
    return search


@pytest.mark.unit
def test_wrapper_modules_are_importable_subclasses():
    assert issubclass(MlSearch, StrategicSearch)
    assert issubclass(MlDetector, PriorityHumanDetector)
    assert issubclass(MlPathPlanning, LoggedAStarPathPlanning)


@pytest.mark.unit
def test_detector_calculate_handles_agent_already_carrying_victim():
    building = Building(10, x=0, y=0)
    refuge = Refuge(20, x=1000, y=0, urn=EntityURN.REFUGE)
    carried = Civilian(100, position=10, hp=7000, damage=40, buriedness=0)
    world = FakeWorldInfo([building, refuge, carried])
    agent = FakeAgentInfo(agent_id=500, position=10, tick=3)
    agent.some_one_on_board = lambda: carried
    detector = make_detector(world, agent)

    result = detector.calculate()
    state = get_runtime_state(agent)

    assert result is detector
    assert detector.get_target_entity_id() == eid(100)
    assert state.phase == "transport"
    assert state.refuge_target == eid(20)
    assert detector._decision_logger.snapshots


@pytest.mark.unit
def test_detector_calculate_no_candidates_switches_to_search_phase():
    building = Building(10, x=0, y=0)
    refuge = Refuge(20, x=1000, y=0, urn=EntityURN.REFUGE)
    world = FakeWorldInfo([building, refuge])
    agent = FakeAgentInfo(agent_id=500, position=10, tick=3)
    agent.some_one_on_board = lambda: None
    detector = make_detector(world, agent)

    detector.calculate()
    state = get_runtime_state(agent)

    assert detector.get_target_entity_id() is None
    assert state.phase == "search"
    assert any(args and args[0] == "detector" for args, _ in detector._decision_logger.snapshots if args)


@pytest.mark.unit
def test_detector_calculate_scores_selects_candidate_and_sets_refuge():
    b10 = Building(10, x=0, y=0)
    b11 = Building(11, x=2000, y=0)
    refuge = Refuge(20, x=3000, y=0, urn=EntityURN.REFUGE)
    c100 = Civilian(100, position=10, hp=9000, damage=30, buriedness=0)
    c101 = Civilian(101, position=11, hp=5000, damage=60, buriedness=3)
    world = FakeWorldInfo([b10, b11, refuge, c100, c101])
    agent = FakeAgentInfo(agent_id=500, position=10, tick=5)
    agent.some_one_on_board = lambda: None
    detector = make_detector(world, agent, cluster_entities=[c100, c101], ml=HybridMl({"100": 0.2, "101": 0.9}))

    detector.calculate()
    state = get_runtime_state(agent)

    assert detector.get_target_entity_id() in {eid(100), eid(101)}
    assert state.phase == "move_to_victim"
    assert state.refuge_target == eid(20)
    assert detector._decision_logger.text


@pytest.mark.unit
def test_detector_selection_order_covers_override_and_block_reasons():
    b10 = Building(10, x=0, y=0)
    b11 = Building(11, x=1000, y=0)
    c100 = Civilian(100, position=10, hp=9000, damage=30, buriedness=0)
    c101 = Civilian(101, position=11, hp=9000, damage=30, buriedness=0)
    world = FakeWorldInfo([b10, b11, c100, c101])
    agent = FakeAgentInfo(agent_id=500, position=10, tick=10)
    detector = make_detector(world, agent, ml=HybridMl())
    state = AgentRuntimeState()
    entity_by_id = {"100": eid(100), "101": eid(101)}
    base_payload = [
        {"candidate_id": "100", "heuristic_score": 1000, "heuristic_component": 1.0, "ml_score": 0.2, "total_trip_distance": 1000, "life_margin": 50, "position_id": "10", "same_position": True, "current_target": False, "blocked": False},
        {"candidate_id": "101", "heuristic_score": 100, "heuristic_component": 0.0, "ml_score": 0.95, "total_trip_distance": 800, "life_margin": 90, "position_id": "11", "same_position": False, "current_target": False, "blocked": False},
    ]

    ordered, selected_by, _, reason, diag = detector._selection_order([dict(p) for p in base_payload], {}, entity_by_id, state, tick=10)
    assert ordered[0]["candidate_id"] == "101"
    assert selected_by == "ml_override"
    assert diag["override_applied"] is True

    detector._ml = HeuristicMl()
    ordered, selected_by, _, reason, diag = detector._selection_order([dict(p) for p in base_payload], {}, entity_by_id, state, tick=10)
    assert selected_by == "heuristic"
    assert diag["override_blocked_reason"] == "ml_inactive"

    detector._ml = HybridMl()
    unreachable = [dict(base_payload[0]), dict(base_payload[1], reachable=False)]
    detector._path_planning = FakePathPlanning(reachable=False)
    ordered, selected_by, _, reason, diag = detector._selection_order(unreachable, {}, entity_by_id, state, tick=10)
    assert diag["override_blocked_reason"] in {"ml_unreachable", "low_confidence", "delta_below_threshold"}


@pytest.mark.unit
def test_detector_scoping_and_nearby_unvisited_buildings():
    source = Building(10, x=0, y=0)
    near = Building(11, x=1000, y=0)
    far = Building(12, x=250000, y=0)
    c100 = Civilian(100, position=10, hp=9000, damage=30, buriedness=0)
    c101 = Civilian(101, position=11, hp=8000, damage=35, buriedness=0)
    c102 = Civilian(102, position=12, hp=3000, damage=80, buriedness=5)
    refuge = Refuge(20, x=0, y=2000, urn=EntityURN.REFUGE)
    world = FakeWorldInfo([source, near, far, refuge, c100, c101, c102])
    agent = FakeAgentInfo(agent_id=500, position=10, tick=6)
    detector = make_detector(world, agent)
    detector._result = eid(102)
    state = AgentRuntimeState()
    state.mark_visited_building(eid(12))

    scoped, info = detector._scoped_candidates([c100, c101, c102], state, tick=6)
    nearby = detector._nearby_unvisited_buildings(eid(10), state)

    assert info["scope_world_count"] == 3
    assert c100 in scoped
    assert c102 in scoped
    assert [b.get_entity_id() for b in nearby] == [eid(11)]


@pytest.mark.unit
def test_search_startup_calculate_selects_reachable_building():
    road = Road(1, x=0, y=0, neighbours=[10, 11])
    b10 = Building(10, x=1000, y=0, neighbours=[1])
    b11 = Building(11, x=2000, y=0, neighbours=[1])
    civ = Civilian(100, position=11, hp=9000, damage=10, buriedness=0)
    world = FakeWorldInfo([road, b10, b11, civ])
    agent = FakeAgentInfo(agent_id=500, position=1, tick=2)
    agent.some_one_on_board = lambda: None
    search = make_search(world, agent, buildings=[b10, b11], cluster_buildings=[b10, b11])

    search.calculate()
    state = get_runtime_state(agent)

    assert search.get_target_entity_id() in {eid(10), eid(11)}
    assert state.search_target == search.get_target_entity_id()
    assert search._decision_logger.snapshots


@pytest.mark.unit
def test_search_calculate_hold_current_building_before_candidate_selection():
    b10 = Building(10, x=0, y=0, neighbours=[1], total_area=90000)
    road = Road(1, x=1000, y=0, neighbours=[10])
    world = FakeWorldInfo([road, b10])
    agent = FakeAgentInfo(agent_id=500, position=10, tick=20)
    agent.some_one_on_board = lambda: None
    search = make_search(world, agent, buildings=[b10], cluster_buildings=[b10])
    state = get_runtime_state(agent)
    state.update_position(eid(1), 18)
    state.update_position(eid(10), 19)
    state.enter_search_building(eid(10), 20, hint_count=0)

    search.calculate()

    assert search.get_target_entity_id() == eid(10)
    assert state.search_target == eid(10)
    assert any("Выбрана" in msg for msg, _ in search._decision_logger.text)


@pytest.mark.unit
def test_search_helper_methods_cover_partition_scope_commitment_and_override():
    road = Road(1, x=0, y=0, neighbours=[10, 11, 12])
    b10 = Building(10, x=1000, y=0, neighbours=[1], total_area=10000)
    b11 = Building(11, x=2000, y=0, neighbours=[1], total_area=80000)
    b12 = Building(12, x=3000, y=0, neighbours=[1], total_area=10000)
    civ = Civilian(100, position=11, hp=9000, damage=20, buriedness=1)
    world = FakeWorldInfo([road, b10, b11, b12, civ])
    agent = FakeAgentInfo(agent_id=500, position=1, tick=30)
    search = make_search(world, agent, buildings=[b10, b11, b12], cluster_buildings=[b10, b11], ml=HybridMl({"10": 0.2, "11": 0.95, "12": 0.1}))
    state = AgentRuntimeState()
    state.mark_visited_building(eid(10))
    search._begin_tick_cache(30)

    cluster_ids, pools = search._partition_candidates(state)
    scope, candidates, reason, forced, cluster_remaining_ratio = search._choose_scope(pools, known_civilians_count=1, tick=30, runtime_state=state)
    state.stationary_ticks = 4
    large_prefiltered = search._prefilter_candidates_for_large_map([b10, b11, b12], state, tick=30)
    features = [search._candidate_features(b, {eid(10), eid(11)}, state, scope, tick=30) for b in candidates]
    for payload in features:
        payload["ml_score"] = {"10": 0.2, "11": 0.95, "12": 0.1}.get(payload["candidate_id"], 0.0)
        payload["final_score"] = payload["ml_score"]
    heuristic_order = sorted(features, key=lambda p: p["heuristic_score"], reverse=True)
    ml_best = max(features, key=lambda p: p["ml_score"])
    ordered, selected_by, diagnostics = search._ml_override_search_order([b10, b11, b12], features, heuristic_order, heuristic_order[0], ml_best, state, 30, True)
    chosen, chosen_reason = search._select_candidate_entity([b10, b11, b12], ordered, state, 30)

    assert pools["cluster_unvisited"] or pools["outside_unvisited"]
    assert candidates
    assert scope
    assert cluster_ids == {eid(10), eid(11)}
    assert large_prefiltered
    assert selected_by in {"heuristic", "ml_override"}
    assert chosen in {eid(10), eid(11), eid(12)}
    assert chosen_reason in {"best_score", "no_reachable_candidate"}

    search._result = eid(11)
    state.set_search_target(eid(11), 28)
    current_payload = next(p for p in features if p["candidate_id"] == "11")
    alt_payload = dict(current_payload, candidate_id="12", first_hop="12", reachable=True, final_score=current_payload["final_score"] + 0.01)
    current_payload["first_hop"] = "11"
    current_payload["reachable"] = True
    kept, keep_reason, info = search._apply_target_commitment(
        chosen=eid(12),
        chosen_reason="heuristic_keep",
        candidate_payload=[current_payload, alt_payload],
        candidates=[b11, b12],
        runtime_state=state,
        tick=30,
        current_area=eid(1),
        current_entity=road,
    )
    assert kept in {eid(11), eid(12)}
    assert "commitment_reason" in info


@pytest.mark.unit
def test_search_building_close_and_loop_helpers():
    b10 = Building(10, x=0, y=0, neighbours=[], total_area=100000)
    refuge = Refuge(20, x=1000, y=0, urn=EntityURN.REFUGE)
    world = FakeWorldInfo([b10, refuge])
    agent = FakeAgentInfo(agent_id=500, position=10, tick=40)
    search = make_search(world, agent, buildings=[b10])
    state = AgentRuntimeState()
    search._result = eid(10)

    info = search._building_close_info(eid(10), b10, state, tick=40, dwell_ticks=99, hint_count=0)
    assert info["exhausted"] is True
    assert search._finalize_building_if_exhausted(eid(10), b10, state, tick=40, reason="unit", dwell_ticks=99, hint_count=0) is True
    assert state.is_blocked("search", eid(10), 41)
    assert search.get_target_entity_id() is None

    search._recent_targets.extend([eid(1), eid(2), eid(1), eid(2), eid(1), eid(2), eid(1), eid(2)])
    assert search._looping_on_local_subset() is True
    state.recent_positions.extend([eid(1), eid(2), eid(3), eid(1), eid(2), eid(3)])
    assert search._position_loop_active(state) is True
    assert search._selection_source("pure_ml_test", False) == "ml"
    assert search._selection_source("hybrid", False) == "hybrid"
    assert search._selection_source("heuristic", True) == "exploratory"

@pytest.mark.unit
def test_search_calculate_full_candidate_ranking_cycle_from_road():
    road = Road(1, x=0, y=0, neighbours=[10, 11, 12])
    b10 = Building(10, x=1000, y=0, neighbours=[1], total_area=10000)
    b11 = Building(11, x=2000, y=0, neighbours=[1], total_area=80000)
    b12 = Building(12, x=3000, y=0, neighbours=[1], total_area=10000)
    civ = Civilian(100, position=11, hp=9000, damage=20, buriedness=1)
    refuge = Refuge(20, x=4000, y=0, urn=EntityURN.REFUGE)
    world = FakeWorldInfo([road, b10, b11, b12, refuge, civ])
    agent = FakeAgentInfo(agent_id=500, position=1, tick=25)
    agent.some_one_on_board = lambda: None
    search = make_search(
        world,
        agent,
        buildings=[b10, b11, b12],
        cluster_buildings=[b10, b11],
        ml=HybridMl({"10": 0.15, "11": 0.95, "12": 0.25}),
    )
    state = get_runtime_state(agent)
    state.update_position(eid(99), 20)
    state.update_position(eid(1), 21)

    search.calculate()

    assert search.get_target_entity_id() in {eid(10), eid(11), eid(12)}
    assert state.search_target == search.get_target_entity_id()
    assert any(args and args[0] == "search" for args, _ in search._decision_logger.snapshots if args)
    assert search._recent_targets
