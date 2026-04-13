from __future__ import annotations

import math
from typing import Iterable, Optional

from rcrscore.entities import Area, Building, Civilian, Entity, EntityID, Human, Refuge, Road
from rcrscore.urn import EntityURN

from .decision_logger import entity_value


SERVICE_BUILDING_CLASS_NAMES = {'AmbulanceCentre', 'FireStation', 'PoliceOffice'}


def is_service_building(entity: Optional[Entity]) -> bool:
    return entity is not None and entity.__class__.__name__ in SERVICE_BUILDING_CLASS_NAMES


def is_searchable_building_entity(entity: Optional[Entity]) -> bool:
    return isinstance(entity, Building) and not isinstance(entity, Refuge) and not is_service_building(entity)


def get_area_id(world_info, entity_id: Optional[EntityID]) -> Optional[EntityID]:
    if entity_id is None:
        return None
    entity = world_info.get_entity(entity_id)
    if isinstance(entity, Area):
        return entity_id
    getter = getattr(world_info, 'get_entity_position_entity_id', None)
    if callable(getter):
        try:
            return getter(entity_id)
        except Exception:
            return None
    return None


def get_xy(entity: Optional[Entity]) -> tuple[Optional[int], Optional[int]]:
    if entity is None:
        return None, None
    get_x = getattr(entity, 'get_x', None)
    get_y = getattr(entity, 'get_y', None)
    if callable(get_x) and callable(get_y):
        try:
            return get_x(), get_y()
        except Exception:
            pass
    getter = getattr(entity, 'get_location', None)
    if callable(getter):
        try:
            result = getter()
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1]
        except Exception:
            pass
    return None, None


def area_distance(world_info, from_entity_id: Optional[EntityID], to_entity_id: Optional[EntityID]) -> float:
    if from_entity_id is None or to_entity_id is None:
        return float('inf')
    from_area = get_area_id(world_info, from_entity_id)
    to_area = get_area_id(world_info, to_entity_id)
    if from_area is None or to_area is None:
        return float('inf')
    try:
        return float(world_info.get_distance(from_area, to_area))
    except Exception:
        return float('inf')


def get_neighbors(area: Area) -> list[EntityID]:
    for name in ('get_neighbors', 'get_neighbours'):
        getter = getattr(area, name, None)
        if callable(getter):
            try:
                values = getter() or []
                result: list[EntityID] = []
                for value in values:
                    if value is None or entity_value(value) == '0':
                        continue
                    result.append(value)
                return result
            except Exception:
                return []
    return []


def build_area_graph(world_info) -> dict[EntityID, list[EntityID]]:
    graph: dict[EntityID, list[EntityID]] = {}
    try:
        entities = world_info.get_entities_of_types([Road, Building])
    except Exception:
        entities = []
    for entity in entities:
        if not isinstance(entity, Area):
            continue
        entity_id = entity.get_entity_id()
        if entity_id is None:
            continue
        graph[entity_id] = get_neighbors(entity)
    return graph


def reconstruct_path(came_from: dict[EntityID, EntityID], current: EntityID) -> list[EntityID]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_distance(world_info, path: list[EntityID]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for index in range(len(path) - 1):
        total += area_distance(world_info, path[index], path[index + 1])
    return total


def summarize_path(world_info, path: list[EntityID]) -> dict[str, float | int]:
    road_nodes = 0
    building_nodes = 0
    for node_id in path:
        entity = world_info.get_entity(node_id)
        if isinstance(entity, Road):
            road_nodes += 1
        elif isinstance(entity, Building):
            building_nodes += 1
    distance_value = path_distance(world_info, path)
    return {
        'node_count': len(path),
        'road_nodes': road_nodes,
        'building_nodes': building_nodes,
        'distance': round(distance_value, 3),
    }


def cluster_entities_for_agent(clustering, agent_entity_id: EntityID) -> list[Entity]:
    try:
        cluster_index = clustering.get_cluster_index(agent_entity_id)
        return list(clustering.get_cluster_entities(cluster_index))
    except Exception:
        return []


def searchable_buildings(world_info, entities: Optional[Iterable[Entity]] = None) -> list[Building]:
    if entities is None:
        try:
            entities = world_info.get_entities_of_types([Building])
        except Exception:
            entities = []
    result: list[Building] = []
    for entity in entities:
        if is_searchable_building_entity(entity):
            result.append(entity)
    return result


def visible_civilians(world_info) -> list[Civilian]:
    try:
        entities = world_info.get_entities_of_types([Civilian])
    except Exception:
        return []
    result: list[Civilian] = []
    for entity in entities:
        if isinstance(entity, Civilian):
            result.append(entity)
    return result


def refuge_entities(world_info) -> list[Refuge]:
    try:
        entities = world_info.get_entities_of_types([Refuge])
    except Exception:
        return []
    return [entity for entity in entities if isinstance(entity, Refuge)]


def nearest_refuge(world_info, from_entity_id: Optional[EntityID]) -> tuple[Optional[EntityID], float]:
    best_id: Optional[EntityID] = None
    best_distance = float('inf')
    for refuge in refuge_entities(world_info):
        refuge_id = refuge.get_entity_id()
        if refuge_id is None:
            continue
        distance_value = area_distance(world_info, from_entity_id, refuge_id)
        if distance_value < best_distance:
            best_distance = distance_value
            best_id = refuge_id
    return best_id, best_distance


def is_transportable_civilian(world_info, entity: Optional[Entity]) -> bool:
    if not isinstance(entity, Civilian):
        return False
    hp = entity.get_hp()
    damage = entity.get_damage()
    buriedness = entity.get_buriedness()
    if hp is None or hp <= 0:
        return False
    if damage is None or damage <= 0:
        return False
    if buriedness is None or buriedness > 0:
        return False
    position_id = entity.get_position()
    if position_id is None:
        return False
    position = world_info.get_entity(position_id)
    if position is None:
        return False
    try:
        urn = position.get_urn()
        if urn in (EntityURN.REFUGE, EntityURN.AMBULANCE_TEAM):
            return False
    except Exception:
        pass
    return True


def is_rescuable_civilian(world_info, entity: Optional[Entity]) -> bool:
    if not isinstance(entity, Civilian):
        return False
    hp = entity.get_hp()
    damage = entity.get_damage()
    buriedness = entity.get_buriedness()
    if hp is None or hp <= 0:
        return False
    if (damage is None or damage <= 0) and (buriedness is None or buriedness <= 0):
        return False
    position_id = entity.get_position()
    if position_id is None:
        return False
    position = world_info.get_entity(position_id)
    if position is None:
        return False
    try:
        urn = position.get_urn()
        if urn in (EntityURN.REFUGE, EntityURN.AMBULANCE_TEAM):
            return False
    except Exception:
        pass
    return True


def civilians_in_building(world_info, building_id: Optional[EntityID], include_stable: bool = False) -> list[Civilian]:
    if building_id is None:
        return []
    result: list[Civilian] = []
    for civilian in visible_civilians(world_info):
        if civilian.get_position() != building_id:
            continue
        hp = civilian.get_hp() or 0
        damage = civilian.get_damage() or 0
        buriedness = civilian.get_buriedness() or 0
        if hp <= 0:
            continue
        if not include_stable and damage <= 0 and buriedness <= 0:
            continue
        result.append(civilian)
    return result


def buried_civilians_in_building(world_info, building_id: Optional[EntityID]) -> list[Civilian]:
    return [c for c in civilians_in_building(world_info, building_id, include_stable=True) if (c.get_buriedness() or 0) > 0 and (c.get_hp() or 0) > 0]


def building_area_value(building: Optional[Building]) -> int:
    if building is None:
        return 0
    for name in ('get_total_area', 'get_ground_area'):
        getter = getattr(building, name, None)
        if callable(getter):
            try:
                value = getter()
                if value is not None:
                    return int(value)
            except Exception:
                pass
    floors_getter = getattr(building, 'get_floors', None)
    edge_getter = getattr(building, 'get_edges', None)
    floors = 1
    edge_count = 0
    if callable(floors_getter):
        try:
            floors = max(int(floors_getter() or 1), 1)
        except Exception:
            floors = 1
    if callable(edge_getter):
        try:
            edge_count = len(edge_getter() or [])
        except Exception:
            edge_count = 0
    return floors * max(edge_count, 4) * 10000

def estimate_life_margin(human: Human, distance_value: float) -> float:
    hp = float(human.get_hp() or 0)
    damage = max(float(human.get_damage() or 0), 1.0)
    buriedness = float(human.get_buriedness() or 0)
    remaining_ticks = hp / damage
    travel_cost = distance_value / 30000.0
    rescue_cost = buriedness * 2.0
    return remaining_ticks - travel_cost - rescue_cost


def urgency_score(human: Human) -> float:
    hp = float(human.get_hp() or 0)
    damage = float(human.get_damage() or 0)
    buriedness = float(human.get_buriedness() or 0)
    return damage * 2.0 + buriedness * 3.0 + max(0.0, 10000.0 - hp) / 1000.0


def count_ambulances_at(world_info, area_id: Optional[EntityID], ignore_agent_id: Optional[EntityID] = None) -> int:
    if area_id is None:
        return 0
    try:
        entities = world_info.get_entities_of_urns([EntityURN.AMBULANCE_TEAM])
    except Exception:
        return 0
    count = 0
    for entity in entities:
        if not isinstance(entity, Human):
            continue
        entity_id = entity.get_entity_id()
        if ignore_agent_id is not None and entity_id == ignore_agent_id:
            continue
        if entity.get_position() == area_id:
            count += 1
    return count


def building_centrality(world_info, building_id: EntityID) -> int:
    entity = world_info.get_entity(building_id)
    if entity is None:
        return 0
    score = 0
    for neighbor_id in get_neighbors(entity):
        neighbor = world_info.get_entity(neighbor_id)
        if isinstance(neighbor, Road):
            score += 1
    return score


def civilians_near_building(world_info, building_id: EntityID, radius: float = 60000.0) -> int:
    count = 0
    for civilian in visible_civilians(world_info):
        position_id = civilian.get_position()
        if area_distance(world_info, position_id, building_id) <= radius:
            count += 1
    return count


def is_search_target_valid(world_info, entity_id: Optional[EntityID]) -> bool:
    if entity_id is None:
        return False
    entity = world_info.get_entity(entity_id)
    return is_searchable_building_entity(entity)


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.inf
