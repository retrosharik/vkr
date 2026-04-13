from __future__ import annotations

from typing import Optional, cast

from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from adf_core_python.core.component.module.complex.search import Search
from rcrscore.entities import Entity, EntityID

from ..util.rescue_module_support import ModuleTrace, cluster_entities_for_agent, get_xy, searchable_building_ids


class CoverageCycleSearch(Search):
    def __init__(
        self,
        agent_info: AgentInfo,
        world_info: WorldInfo,
        scenario_info: ScenarioInfo,
        module_manager: ModuleManager,
        develop_data: DevelopData,
    ) -> None:
        super().__init__(agent_info, world_info, scenario_info, module_manager, develop_data)
        self._clustering: Clustering = cast(
            Clustering,
            module_manager.get_module(
                "DefaultSearch.Clustering",
                "adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering",
            ),
        )
        self.register_sub_module(self._clustering)
        self._visited: set[EntityID] = set()
        self._ordered_targets: list[EntityID] = []
        self._cursor: int = 0
        self._result: Optional[EntityID] = None
        self._trace = ModuleTrace(agent_info, world_info, self.__class__.__name__)

    def _rebuild_targets(self) -> None:
        me = self._agent_info.get_entity_id()
        cluster_entities = cluster_entities_for_agent(self._clustering, me)
        candidates = searchable_building_ids(self._world_info, cluster_entities)
        if not candidates:
            candidates = searchable_building_ids(self._world_info)
        def sort_key(entity_id: EntityID) -> tuple[int, int, int]:
            entity = self._world_info.get_entity(entity_id)
            x, y = get_xy(entity)
            return (y or 0, x or 0, entity_id.get_value())
        self._ordered_targets = sorted(candidates, key=sort_key)
        self._cursor = 0

    def update_info(self, message_manager: MessageManager) -> Search:
        super().update_info(message_manager)
        self._trace.log_position_if_changed()
        current = self._agent_info.get_position_entity_id()
        if current is not None:
            self._visited.add(current)
            if self._result == current:
                self._result = None
        return self

    def calculate(self) -> Search:
        try:
            if not self._ordered_targets:
                self._rebuild_targets()
            if not self._ordered_targets:
                self._result = None
                self._trace.log("search_target_selected target=None reason=no_searchable_building")
                return self
            checked = 0
            while checked < len(self._ordered_targets):
                candidate = self._ordered_targets[self._cursor % len(self._ordered_targets)]
                self._cursor += 1
                checked += 1
                if candidate not in self._visited:
                    self._result = candidate
                    self._trace.log_target("search_target", self._result, f"strategy=coverage_cycle cursor={self._cursor}")
                    return self
            self._visited.clear()
            self._rebuild_targets()
            if not self._ordered_targets:
                self._result = None
                self._trace.log("search_target_selected target=None reason=no_rebuilt_targets")
                return self
            self._result = self._ordered_targets[self._cursor % len(self._ordered_targets)]
            self._cursor += 1
            self._trace.log_target("search_target", self._result, f"strategy=coverage_cycle_reset cursor={self._cursor}")
            return self
        except Exception as exc:
            self._trace.error("calculate", exc)
            self._result = None
            return self

    def get_target_entity_id(self) -> Optional[EntityID]:
        return self._result
