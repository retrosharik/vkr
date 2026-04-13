from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET

SCENARIO_NS = 'urn:roborescue:map:scenario'
NS = {'scenario': SCENARIO_NS}
LOCATION = f'{{{SCENARIO_NS}}}location'
BED_CAPACITY = f'{{{SCENARIO_NS}}}bedCapacity'
ET.register_namespace('scenario', SCENARIO_NS)


@dataclass(frozen=True)
class RefugeEntry:
    location: int
    bed_capacity: int


@dataclass(frozen=True)
class ScenarioBase:
    refuges: List[RefugeEntry]
    ambulance_locations: List[int]
    civilian_locations: List[int]


def read_base_scenario(path: Path) -> ScenarioBase:
    root = ET.parse(path).getroot()
    refuges: List[RefugeEntry] = []
    ambulance_locations: List[int] = []
    civilian_locations: List[int] = []

    for refuge in root.findall('scenario:refuge', NS):
        location = refuge.get(LOCATION)
        bed_capacity = refuge.get(BED_CAPACITY, '1')
        if location is None:
            continue
        refuges.append(RefugeEntry(location=int(location), bed_capacity=int(bed_capacity)))

    for amb in root.findall('scenario:ambulanceteam', NS):
        location = amb.get(LOCATION)
        if location is not None:
            ambulance_locations.append(int(location))

    for civ in root.findall('scenario:civilian', NS):
        location = civ.get(LOCATION)
        if location is not None:
            civilian_locations.append(int(location))

    return ScenarioBase(refuges=refuges, ambulance_locations=ambulance_locations, civilian_locations=civilian_locations)


def write_scenario(path: Path, refuges: List[RefugeEntry], ambulance_location: int, civilian_locations: List[int]) -> None:
    root = ET.Element(f'{{{SCENARIO_NS}}}scenario')

    for refuge in refuges:
        ET.SubElement(
            root,
            f'{{{SCENARIO_NS}}}refuge',
            {LOCATION: str(refuge.location), BED_CAPACITY: str(refuge.bed_capacity)},
        )

    ET.SubElement(root, f'{{{SCENARIO_NS}}}ambulanceteam', {LOCATION: str(ambulance_location)})

    for location in civilian_locations:
        ET.SubElement(root, f'{{{SCENARIO_NS}}}civilian', {LOCATION: str(location)})

    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding='UTF-8', xml_declaration=True)
