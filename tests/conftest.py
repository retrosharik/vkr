from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Iterable


def _project_root() -> Path:
    env_root = os.environ.get("VKR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    return Path(__file__).resolve().parents[1]


def _agent_root() -> Path:
    env_root = os.environ.get("BASERESCUEAGENT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return _project_root() / "BaseRescueAgent"


PROJECT_ROOT = _project_root()
AGENT_ROOT = _agent_root()
SRC_ROOT = AGENT_ROOT / "src"

if not AGENT_ROOT.exists():
    raise RuntimeError(
        "BaseRescueAgent directory was not found. Run tests from the project root "
        "or set BASERESCUEAGENT_ROOT=/path/to/BaseRescueAgent."
    )

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FakeEntityID:
    def __init__(self, value: int | str):
        self._value = int(value)

    def get_value(self) -> int:
        return self._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        getter = getattr(other, "get_value", None)
        if callable(getter):
            return self._value == int(getter())
        return False

    def __repr__(self) -> str:
        return f"EntityID({self._value})"


class FakeEntity:
    def __init__(self, entity_id: FakeEntityID | int, *, x: int = 0, y: int = 0, urn: object = None):
        self._entity_id = entity_id if isinstance(entity_id, FakeEntityID) else FakeEntityID(entity_id)
        self._x = x
        self._y = y
        self._urn = urn

    def get_entity_id(self):
        return self._entity_id

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_location(self):
        return self._x, self._y

    def get_urn(self):
        return self._urn


class FakeArea(FakeEntity):
    def __init__(self, entity_id: FakeEntityID | int, *, neighbours: Iterable[FakeEntityID | int] = (), **kwargs):
        super().__init__(entity_id, **kwargs)
        self._neighbours = [n if isinstance(n, FakeEntityID) else FakeEntityID(n) for n in neighbours]

    def get_neighbors(self):
        return list(self._neighbours)

    def get_neighbours(self):
        return list(self._neighbours)


class FakeRoad(FakeArea):
    pass


class FakeBuilding(FakeArea):
    def __init__(self, entity_id: FakeEntityID | int, *, total_area: int = 10000, **kwargs):
        super().__init__(entity_id, **kwargs)
        self._total_area = total_area

    def get_total_area(self):
        return self._total_area

    def get_ground_area(self):
        return self._total_area

    def get_floors(self):
        return 1

    def get_edges(self):
        return [1, 2, 3, 4]


class FakeRefuge(FakeBuilding):
    pass


class FakeHuman(FakeEntity):
    def __init__(self, entity_id: FakeEntityID | int, *, position=None, hp=10000, damage=0, buriedness=0, **kwargs):
        super().__init__(entity_id, **kwargs)
        self._position = position if position is None or isinstance(position, FakeEntityID) else FakeEntityID(position)
        self._hp = hp
        self._damage = damage
        self._buriedness = buriedness

    def get_position(self):
        return self._position

    def get_hp(self):
        return self._hp

    def get_damage(self):
        return self._damage

    def get_buriedness(self):
        return self._buriedness


class FakeCivilian(FakeHuman):
    pass


class FakeWorldInfo:
    def __init__(self, entities: Iterable[FakeEntity] = ()): 
        self.entities = {e.get_entity_id(): e for e in entities}

    def get_entity(self, entity_id):
        return self.entities.get(entity_id)

    def get_entity_position_entity_id(self, entity_id):
        entity = self.get_entity(entity_id)
        getter = getattr(entity, "get_position", None)
        return getter() if callable(getter) else None

    def get_entities_of_types(self, classes):
        return [e for e in self.entities.values() if isinstance(e, tuple(classes))]

    def get_entities_of_urns(self, urns):
        return [e for e in self.entities.values() if getattr(e, "get_urn", lambda: None)() in set(urns)]

    def get_distance(self, a, b):
        ea = self.get_entity(a)
        eb = self.get_entity(b)
        if ea is None or eb is None:
            raise KeyError("unknown entity")
        return ((ea.get_x() - eb.get_x()) ** 2 + (ea.get_y() - eb.get_y()) ** 2) ** 0.5


class FakeAgentInfo:
    def __init__(self, agent_id=1, position=1, tick=0):
        self._agent_id = FakeEntityID(agent_id)
        self._position = position if isinstance(position, FakeEntityID) else FakeEntityID(position)
        self._tick = tick

    def get_entity_id(self):
        return self._agent_id

    def get_position_entity_id(self):
        return self._position

    def set_position_entity_id(self, position):
        self._position = position if isinstance(position, FakeEntityID) else FakeEntityID(position)

    def get_time(self):
        return self._tick

    def set_time(self, tick):
        self._tick = int(tick)


class FakeBaseModule:
    def __init__(self, agent_info=None, world_info=None, scenario_info=None, module_manager=None, develop_data=None):
        self._agent_info = agent_info
        self._world_info = world_info
        self._scenario_info = scenario_info
        self._module_manager = module_manager
        self._develop_data = develop_data

    def update_info(self, *args, **kwargs):
        return self

    def calculate(self, *args, **kwargs):
        return self


class FakeModuleManager:
    def get_module(self, *args, **kwargs):
        return None


class FakeEntityURN:
    REFUGE = "REFUGE"
    AMBULANCE_TEAM = "AMBULANCE_TEAM"


def _install_module(path: str, **attrs) -> None:
    module = types.ModuleType(path)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[path] = module


def _ensure_package(path: str):
    module = sys.modules.get(path)
    if module is None:
        module = types.ModuleType(path)
        sys.modules[path] = module
    if not hasattr(module, "__path__"):
        module.__path__ = []
    return module


for pkg in [
    "rcrscore",
    "adf_core_python",
    "adf_core_python.core",
    "adf_core_python.core.agent",
    "adf_core_python.core.agent.communication",
    "adf_core_python.core.agent.develop",
    "adf_core_python.core.agent.info",
    "adf_core_python.core.agent.module",
    "adf_core_python.core.component",
    "adf_core_python.core.component.module",
    "adf_core_python.core.component.module.algorithm",
    "adf_core_python.core.component.module.complex",
]:
    _ensure_package(pkg)

_install_module(
    "rcrscore.entities",
    EntityID=FakeEntityID,
    Entity=FakeEntity,
    Area=FakeArea,
    Building=FakeBuilding,
    Road=FakeRoad,
    Refuge=FakeRefuge,
    Human=FakeHuman,
    Civilian=FakeCivilian,
)
_install_module("rcrscore.urn", EntityURN=FakeEntityURN)

_install_module("adf_core_python.core.agent.communication.message_manager", MessageManager=object)
_install_module("adf_core_python.core.agent.develop.develop_data", DevelopData=object)
_install_module("adf_core_python.core.agent.info.agent_info", AgentInfo=FakeAgentInfo)
_install_module("adf_core_python.core.agent.info.scenario_info", ScenarioInfo=object)
_install_module("adf_core_python.core.agent.info.world_info", WorldInfo=FakeWorldInfo)
_install_module("adf_core_python.core.agent.module.module_manager", ModuleManager=FakeModuleManager)
_install_module("adf_core_python.core.component.module.algorithm.path_planning", PathPlanning=FakeBaseModule)
_install_module("adf_core_python.core.component.module.algorithm.clustering", Clustering=object)
_install_module("adf_core_python.core.component.module.complex.human_detector", HumanDetector=FakeBaseModule)
_install_module("adf_core_python.core.component.module.complex.search", Search=FakeBaseModule)
