from __future__ import annotations

from collections import deque
from typing import Optional

from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from rcrscore.entities import EntityID

from ..util.rescue_module_support import ModuleTrace, build_area_graph, get_area_id, path_distance, reconstruct_path


class BFSPathPlanning(PathPlanning):
    def __init__(
        self,
        agent_info: AgentInfo,
        world_info: WorldInfo,
        scenario_info: ScenarioInfo,
        module_manager: ModuleManager,
        develop_data: DevelopData,
    ) -> None:
        super().__init__(agent_info, world_info, scenario_info, module_manager, develop_data)
        self._graph = build_area_graph(world_info)
        self._trace = ModuleTrace(agent_info, world_info, self.__class__.__name__)
        self._last_signature: Optional[tuple[str, str, int]] = None

    def update_info(self, message_manager: MessageManager) -> PathPlanning:
        super().update_info(message_manager)
        self._trace.log_position_if_changed()
        return self

    def calculate(self) -> PathPlanning:
        self._trace.log_position_if_changed()
        return self

    def _normalize(self, entity_id: EntityID) -> Optional[EntityID]:
        return get_area_id(self._world_info, entity_id)

    def get_path(self, from_entity_id: EntityID, to_entity_id: EntityID) -> list[EntityID]:
        try:
            start = self._normalize(from_entity_id)
            goal = self._normalize(to_entity_id)
            if start is None or goal is None:
                return []
            if start == goal:
                return [start]
            queue: deque[EntityID] = deque([start])
            came_from: dict[EntityID, EntityID] = {}
            seen: set[EntityID] = {start}
            while queue:
                current = queue.popleft()
                for neighbor in self._graph.get(current, []):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    came_from[neighbor] = current
                    if neighbor == goal:
                        path = reconstruct_path(came_from, goal)
                        self._log_path(start, goal, path)
                        return path
                    queue.append(neighbor)
            return []
        except Exception as exc:
            self._trace.error("get_path", exc)
            return []

    def get_path_to_multiple_destinations(self, from_entity_id: EntityID, destination_entity_ids: set[EntityID]) -> list[EntityID]:
        best_path: list[EntityID] = []
        best_distance: Optional[float] = None
        for destination in destination_entity_ids:
            path = self.get_path(from_entity_id, destination)
            if not path:
                continue
            distance_value = path_distance(self._world_info, path)
            if best_distance is None or distance_value < best_distance:
                best_distance = distance_value
                best_path = path
        return best_path

    def get_distance(self, from_entity_id: EntityID, to_entity_id: EntityID) -> float:
        return path_distance(self._world_info, self.get_path(from_entity_id, to_entity_id))

    def _log_path(self, start: EntityID, goal: EntityID, path: list[EntityID]) -> None:
        signature = (str(start), str(goal), len(path))
        if signature != self._last_signature:
            self._last_signature = signature
            self._trace.log_path(start, goal, path)
