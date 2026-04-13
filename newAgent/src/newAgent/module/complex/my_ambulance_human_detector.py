from typing import Set
import os

from rcrscore.entities import Civilian, EntityID

from src.newAgent.module.complex.sample_human_detector import SampleHumanDetector
from src.newAgent.module.util.ambulance_file_logger import AmbulanceFileLogger


class MyAmbulanceHumanDetector(SampleHumanDetector):
    def __init__(self, agent_info, world_info, scenario_info, module_manager, develop_data):
        super().__init__(agent_info, world_info, scenario_info, module_manager, develop_data)
        

        map_name = "unknown_map"
        self.logger = AmbulanceFileLogger(map_name)
        self._logged_humans: Set[EntityID] = set()
        self._first_calculate = True
        self._logged_end = False
        self._max_cycle = scenario_info.get_value("kernel.timesteps", 999999)

    def calculate(self):
        old_result = self._result


        super().calculate()

        current_time = self._agent_info.get_time()

        if self._first_calculate:
            self.logger.log_start(current_time)
            self._first_calculate = False

        if (self._result is not None and 
            self._result != old_result and 
            self._result not in self._logged_humans):
            
            human = self._world_info.get_entity(self._result)
            if human and isinstance(human, Civilian):
                x = getattr(human, "get_x", lambda: 0)()
                y = getattr(human, "get_y", lambda: 0)()
                self.logger.log_human_detected(
                    current_time,
                    int(self._result.get_value()),
                    x,
                    y
                )
                self._logged_humans.add(self._result)

        if not self._logged_end and current_time >= self._max_cycle - 5:
            self.logger.log_end(current_time)
            self._logged_end = True

        return self
