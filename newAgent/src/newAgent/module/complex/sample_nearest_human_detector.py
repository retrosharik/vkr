from __future__ import annotations

from typing import Optional, cast

from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from adf_core_python.core.component.module.complex.human_detector import HumanDetector
from rcrscore.entities import Civilian, Entity, EntityID, Human

from ..util.rescue_module_support import ModuleTrace, area_distance, cluster_entities_for_agent, entity_value, is_transportable_civilian


class SampleNearestHumanDetector(HumanDetector):
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
                "DefaultHumanDetector.Clustering",
                "adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering",
            ),
        )
        self.register_sub_module(self._clustering)
        self._result: Optional[EntityID] = None
        self._trace = ModuleTrace(agent_info, world_info, self.__class__.__name__)

    def update_info(self, message_manager: MessageManager) -> HumanDetector:
        super().update_info(message_manager)
        self._trace.log_position_if_changed()
        return self

    def calculate(self) -> HumanDetector:
        try:
            self._trace.log_position_if_changed()
            carried: Optional[Human] = self._agent_info.some_one_on_board()
            if carried is not None:
                self._result = carried.get_entity_id()
                self._trace.log_target("human_target", self._result, "state=carrying")
                return self
            if self._result is not None:
                entity = self._world_info.get_entity(self._result)
                if not is_transportable_civilian(self._world_info, entity):
                    self._result = None
            if self._result is None:
                self._result = self._select_target()
            return self
        except Exception as exc:
            self._trace.error("calculate", exc)
            self._result = None
            return self

    def _select_target(self) -> Optional[EntityID]:
        me = self._agent_info.get_entity_id()
        cluster_entities = cluster_entities_for_agent(self._clustering, me)
        cluster_targets = [entity for entity in cluster_entities if isinstance(entity, Civilian) and is_transportable_civilian(self._world_info, entity)]
        world_targets = [
            entity
            for entity in self._world_info.get_entities_of_types([Civilian])
            if isinstance(entity, Civilian) and is_transportable_civilian(self._world_info, entity)
        ]
        candidates = cluster_targets if cluster_targets else world_targets
        if not candidates:
            self._trace.log("human_target_selected target=None reason=no_transportable_civilian")
            return None
        target = min(candidates, key=lambda entity: area_distance(self._world_info, me, entity.get_entity_id()))
        extra = f"strategy=nearest distance={area_distance(self._world_info, me, target.get_entity_id()):.1f}"
        self._trace.log_target("human_target", target.get_entity_id(), extra)
        return target.get_entity_id()

    def get_target_entity_id(self) -> Optional[EntityID]:
        return self._result
