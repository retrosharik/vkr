
from __future__ import annotations

import logging
import math
import time
import traceback
from pathlib import Path
from typing import Iterable, Optional

from rcrscore.entities import Area, Building, Civilian, Entity, EntityID, Human, Refuge, Road
from rcrscore.urn import EntityURN


def entity_value(entity_id: Optional[EntityID]) -> str:
    if entity_id is None:
        return "None"
    getter = getattr(entity_id, "get_value", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return str(entity_id)
    return str(entity_id)


def get_area_id(world_info, entity_id: Optional[EntityID]) -> Optional[EntityID]:
    if entity_id is None:
        return None
    entity = world_info.get_entity(entity_id)
    if isinstance(entity, Area):
        return entity_id
    try:
        return world_info.get_entity_position_entity_id(entity_id)
    except Exception:
        return None


def get_xy(entity: Optional[Entity]) -> tuple[Optional[int], Optional[int]]:
    if entity is None:
        return None, None
    get_x = getattr(entity, "get_x", None)
    get_y = getattr(entity, "get_y", None)
    if callable(get_x) and callable(get_y):
        try:
            return get_x(), get_y()
        except Exception:
            pass
    get_location = getattr(entity, "get_location", None)
    if callable(get_location):
        try:
            location = get_location()
            if isinstance(location, tuple) and len(location) == 2:
                return location[0], location[1]
        except Exception:
            pass
    return None, None


def get_neighbors(area: Area) -> list[EntityID]:
    for name in ("get_neighbors", "get_neighbours"):
        getter = getattr(area, name, None)
        if callable(getter):
            try:
                neighbors = getter() or []
                result: list[EntityID] = []
                for neighbor in neighbors:
                    if neighbor is None:
                        continue
                    if entity_value(neighbor) == "0":
                        continue
                    result.append(neighbor)
                return result
            except Exception:
                return []
    return []


def area_distance(world_info, from_entity_id: Optional[EntityID], to_entity_id: Optional[EntityID]) -> float:
    if from_entity_id is None or to_entity_id is None:
        return float("inf")
    from_area_id = get_area_id(world_info, from_entity_id)
    to_area_id = get_area_id(world_info, to_entity_id)
    if from_area_id is None or to_area_id is None:
        return float("inf")
    try:
        return float(world_info.get_distance(from_area_id, to_area_id))
    except Exception:
        return float("inf")


def format_entity(world_info, entity_id: Optional[EntityID]) -> str:
    if entity_id is None:
        return "None"
    entity = world_info.get_entity(entity_id)
    urn = None
    if entity is not None:
        try:
            urn = entity.get_urn().name
        except Exception:
            urn = entity.__class__.__name__
    x, y = get_xy(entity)
    return f"id={entity_value(entity_id)} urn={urn} x={x} y={y}"


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


def transportable_civilians(world_info) -> list[Civilian]:
    result: list[Civilian] = []
    try:
        entities = world_info.get_entities_of_types([Civilian])
    except Exception:
        return result
    for entity in entities:
        if isinstance(entity, Civilian) and is_transportable_civilian(world_info, entity):
            result.append(entity)
    return result


def searchable_building_ids(world_info, entities: Optional[Iterable[Entity]] = None) -> list[EntityID]:
    if entities is None:
        try:
            entities = world_info.get_entities_of_types([Building])
        except Exception:
            entities = []
    result: list[EntityID] = []
    for entity in entities:
        if not isinstance(entity, Building):
            continue
        if isinstance(entity, Refuge):
            continue
        entity_id = entity.get_entity_id()
        if entity_id is None:
            continue
        result.append(entity_id)
    return result


def estimate_life_margin(human: Human, distance_value: float) -> float:
    hp = human.get_hp() or 0
    damage = max(human.get_damage() or 0, 1)
    buriedness = human.get_buriedness() or 0
    remaining = hp / damage
    travel = distance_value / 30000.0
    rescue_delay = buriedness * 2.0
    return remaining - travel - rescue_delay


def urgency_score(human: Human) -> float:
    hp = human.get_hp() or 0
    damage = human.get_damage() or 0
    buriedness = human.get_buriedness() or 0
    return damage * 2.0 + buriedness * 3.0 + max(0.0, 10000.0 - hp) / 1000.0


def count_ambulances_at(world_info, area_id: Optional[EntityID], ignore_agent_id: Optional[EntityID] = None) -> int:
    if area_id is None:
        return 0
    count = 0
    try:
        entities = world_info.get_entities_of_urns([EntityURN.AMBULANCE_TEAM])
    except Exception:
        entities = []
    for entity in entities:
        if not isinstance(entity, Human):
            continue
        entity_id = entity.get_entity_id()
        if ignore_agent_id is not None and entity_id == ignore_agent_id:
            continue
        if entity.get_position() == area_id:
            count += 1
    return count


def blockades_penalty(world_info, area_id: Optional[EntityID]) -> float:
    if area_id is None:
        return 0.0
    area = world_info.get_entity(area_id)
    if not isinstance(area, Area):
        return 0.0
    try:
        blockades = world_info.get_blockades(area)
        return float(len(blockades)) * 5000.0
    except Exception:
        return 0.0


def road_bias_penalty(world_info, entity_id: Optional[EntityID], destination_id: Optional[EntityID]) -> float:
    if entity_id is None:
        return 0.0
    if destination_id is not None and entity_id == destination_id:
        return 0.0
    entity = world_info.get_entity(entity_id)
    if entity is None:
        return 0.0
    if isinstance(entity, Road):
        return 0.0
    if isinstance(entity, Building):
        return 1000.0
    return 0.0


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
    total_path: list[EntityID] = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path


def path_distance(world_info, path: list[EntityID]) -> float:
    if len(path) < 2:
        return 0.0
    distance_value = 0.0
    for index in range(len(path) - 1):
        distance_value += area_distance(world_info, path[index], path[index + 1])
    return distance_value


def cluster_entities_for_agent(clustering, agent_entity_id: EntityID) -> list[Entity]:
    try:
        cluster_index = clustering.get_cluster_index(agent_entity_id)
        return list(clustering.get_cluster_entities(cluster_index))
    except Exception:
        return []


class ModuleTrace:
    def __init__(self, agent_info, world_info, module_name: str) -> None:
        self._agent_info = agent_info
        self._world_info = world_info
        self._module_name = module_name
        self._start_wall = time.monotonic()
        self._last_position_id: Optional[EntityID] = None
        self._logger = self._create_logger()
        self.log(f"module_loaded name={module_name}")

    def _project_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "main.py").exists() and (parent / "config").exists():
                return parent
        return Path.cwd()

    def _create_logger(self) -> logging.Logger:
        project_root = self._project_root()
        result_dir = project_root / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        agent_id = entity_value(getattr(self._agent_info, "get_entity_id")())
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logger_name = f"{self._module_name}.{agent_id}.{timestamp}.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(message)s")
        file_handler = logging.FileHandler(
            result_dir / f"{self._module_name}_{agent_id}_{timestamp}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _prefix(self) -> str:
        tick = 0
        try:
            tick = int(self._agent_info.get_time())
        except Exception:
            tick = 0
        position_id = None
        try:
            position_id = self._agent_info.get_position_entity_id()
        except Exception:
            position_id = None
        position = self._world_info.get_entity(position_id) if position_id is not None else None
        x, y = get_xy(position)
        wall_elapsed = time.monotonic() - self._start_wall
        agent_id = entity_value(getattr(self._agent_info, "get_entity_id")())
        return (
            f"[module={self._module_name} agent={agent_id} tick={tick} "
            f"sim_elapsed={tick} wall_elapsed={wall_elapsed:.3f}s "
            f"area={entity_value(position_id)} x={x} y={y}]"
        )

    def log(self, message: str) -> None:
        self._logger.info(f"{self._prefix()} {message}")

    def error(self, where: str, exc: Exception) -> None:
        self._logger.error(f"{self._prefix()} error where={where} type={exc.__class__.__name__} message={exc}")
        self._logger.error(traceback.format_exc())

    def log_position_if_changed(self) -> None:
        try:
            position_id = self._agent_info.get_position_entity_id()
        except Exception:
            position_id = None
        if position_id != self._last_position_id:
            self._last_position_id = position_id
            position = self._world_info.get_entity(position_id) if position_id is not None else None
            x, y = get_xy(position)
            self.log(f"agent_position_changed area={entity_value(position_id)} x={x} y={y}")

    def log_target(self, target_kind: str, target_id: Optional[EntityID], extra: str = "") -> None:
        if target_id is None:
            self.log(f"{target_kind}_selected target=None")
            return
        self.log(f"{target_kind}_selected {format_entity(self._world_info, target_id)} {extra}".strip())

    def log_path(self, from_id: Optional[EntityID], to_id: Optional[EntityID], path: list[EntityID]) -> None:
        distance_value = path_distance(self._world_info, path)
        next_id = path[1] if len(path) > 1 else (path[0] if path else None)
        self.log(
            f"path_computed from={entity_value(from_id)} to={entity_value(to_id)} "
            f"nodes={len(path)} distance={distance_value:.1f} next={format_entity(self._world_info, next_id)}"
        )
