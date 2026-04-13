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
from rcrscore.entities import EntityID

from ..util.rescue_module_support import ModuleTrace, area_distance, cluster_entities_for_agent, entity_value, searchable_building_ids


class ClusterNearestSearch(Search):
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
        self._result: Optional[EntityID] = None
        self._trace = ModuleTrace(agent_info, world_info, self.__class__.__name__)

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
            self._trace.log_position_if_changed()
            if self._result is not None and self._result not in self._visited:
                self._trace.log_target("search_target", self._result, "state=kept")
                return self
            me = self._agent_info.get_entity_id()
            cluster_entities = cluster_entities_for_agent(self._clustering, me)
            cluster_ids = [entity_id for entity_id in searchable_building_ids(self._world_info, cluster_entities) if entity_id not in self._visited]
            world_ids = [entity_id for entity_id in searchable_building_ids(self._world_info) if entity_id not in self._visited]
            candidates = cluster_ids if cluster_ids else world_ids
            if not candidates:
                self._visited.clear()
                candidates = searchable_building_ids(self._world_info)
            if not candidates:
                self._result = None
                self._trace.log("search_target_selected target=None reason=no_searchable_building")
                return self
            self._result = min(candidates, key=lambda entity_id: area_distance(self._world_info, me, entity_id))
            self._trace.log_target("search_target", self._result, f"strategy=cluster_nearest remaining={len(candidates)}")
            return self
        except Exception as exc:
            self._trace.error("calculate", exc)
            self._result = None
            return self

    def get_target_entity_id(self) -> Optional[EntityID]:
        return self._result
