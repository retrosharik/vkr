from __future__ import annotations

import math

import pytest
from rcrscore.entities import Building, Civilian, EntityID, Refuge, Road
from rcrscore.urn import EntityURN

from BaseRescueAgent.module.util import rescue_support as rs
from tests.conftest import FakeCivilian, FakeEntity, FakeHuman, FakeWorldInfo


def eid(value: int) -> EntityID:
    return EntityID(value)


@pytest.mark.unit
def test_build_area_graph_filters_zero_neighbors_and_summarizes_path():
    r1 = Road(1, x=0, y=0, neighbours=[2, 0])
    r2 = Road(2, x=1000, y=0, neighbours=[1, 3])
    b3 = Building(3, x=2000, y=0, neighbours=[2], total_area=12345)
    world = FakeWorldInfo([r1, r2, b3])

    graph = rs.build_area_graph(world)
    path = rs.reconstruct_path({eid(2): eid(1), eid(3): eid(2)}, eid(3))
    summary = rs.summarize_path(world, path)

    assert graph[eid(1)] == [eid(2)]
    assert path == [eid(1), eid(2), eid(3)]
    assert rs.path_distance(world, path) == pytest.approx(2000.0)
    assert summary == {"node_count": 3, "road_nodes": 2, "building_nodes": 1, "distance": 2000.0}


@pytest.mark.unit
def test_searchable_buildings_and_visible_civilians_filter_expected_entities():
    class AmbulanceCentre(Building):
        pass

    regular = Building(10)
    refuge = Refuge(11)
    service = AmbulanceCentre(12)
    road = Road(13)
    civilian = Civilian(100, position=10, hp=9000, damage=10, buriedness=0)
    world = FakeWorldInfo([regular, refuge, service, road, civilian])

    assert rs.is_searchable_building_entity(regular)
    assert not rs.is_searchable_building_entity(refuge)
    assert not rs.is_searchable_building_entity(service)
    assert rs.searchable_buildings(world) == [regular]
    assert rs.visible_civilians(world) == [civilian]
    assert rs.refuge_entities(world) == [refuge]


@pytest.mark.unit
def test_civilian_rescue_and_transport_filters_cover_key_cases():
    building = Building(10)
    refuge = Refuge(11, urn=EntityURN.REFUGE)
    buried = Civilian(100, position=10, hp=9000, damage=1, buriedness=5)
    transportable = Civilian(101, position=10, hp=9000, damage=10, buriedness=0)
    stable = Civilian(102, position=10, hp=9000, damage=0, buriedness=0)
    in_refuge = Civilian(103, position=11, hp=9000, damage=10, buriedness=0)
    dead = Civilian(104, position=10, hp=0, damage=10, buriedness=0)
    world = FakeWorldInfo([building, refuge, buried, transportable, stable, in_refuge, dead])

    assert rs.is_rescuable_civilian(world, buried)
    assert rs.is_rescuable_civilian(world, transportable)
    assert not rs.is_rescuable_civilian(world, stable)
    assert not rs.is_rescuable_civilian(world, in_refuge)
    assert not rs.is_rescuable_civilian(world, dead)

    assert rs.is_transportable_civilian(world, transportable)
    assert not rs.is_transportable_civilian(world, buried)
    assert not rs.is_transportable_civilian(world, stable)

    assert rs.civilians_in_building(world, eid(10)) == [buried, transportable]
    assert rs.buried_civilians_in_building(world, eid(10)) == [buried]


@pytest.mark.unit
def test_nearest_refuge_centrality_life_urgency_and_safe_float():
    b1 = Building(1, x=0, y=0, neighbours=[2, 3])
    road = Road(2, x=1000, y=0)
    refuge_far = Refuge(3, x=5000, y=0)
    refuge_near = Refuge(4, x=1500, y=0)
    civilian = Civilian(100, position=1, hp=8000, damage=20, buriedness=2)
    world = FakeWorldInfo([b1, road, refuge_far, refuge_near, civilian])

    refuge_id, distance = rs.nearest_refuge(world, eid(1))
    assert refuge_id == eid(4)
    assert distance == pytest.approx(1500.0)
    assert rs.building_centrality(world, eid(1)) == 1
    assert rs.civilians_near_building(world, eid(1), radius=1.0) == 1
    assert rs.estimate_life_margin(civilian, 30000.0) == pytest.approx(395.0)
    assert rs.urgency_score(civilian) == pytest.approx(2 * 20 + 3 * 2 + 2)
    assert rs.safe_float("12.5") == 12.5
    assert math.isinf(rs.safe_float(object()))


@pytest.mark.unit
def test_count_ambulances_at_ignores_current_agent():
    area = Road(1)
    current = FakeHuman(200, position=1, urn=EntityURN.AMBULANCE_TEAM)
    teammate = FakeHuman(201, position=1, urn=EntityURN.AMBULANCE_TEAM)
    elsewhere = FakeHuman(202, position=2, urn=EntityURN.AMBULANCE_TEAM)
    world = FakeWorldInfo([area, Road(2), current, teammate, elsewhere])

    assert rs.count_ambulances_at(world, eid(1), ignore_agent_id=eid(200)) == 1
    assert rs.count_ambulances_at(world, eid(1), ignore_agent_id=None) == 2
