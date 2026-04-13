from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import math
import xml.etree.ElementTree as ET

NS = {
    'rcr': 'urn:roborescue:map:gml',
    'gml': 'http://www.opengis.net/gml',
    'xlink': 'http://www.w3.org/1999/xlink',
}
GML_ID = '{http://www.opengis.net/gml}id'
XLINK_HREF = '{http://www.w3.org/1999/xlink}href'


@dataclass(frozen=True)
class BuildingInfo:
    entity_id: int
    centroid: Tuple[float, float]


@dataclass(frozen=True)
class MapInfo:
    buildings: List[BuildingInfo]


@dataclass(frozen=True)
class MapStats:
    building_count: int


def _parse_nodes(root: ET.Element) -> Dict[str, Tuple[float, float]]:
    nodes: Dict[str, Tuple[float, float]] = {}
    for node in root.findall('.//rcr:nodelist/gml:Node', NS):
        node_id = node.get(GML_ID)
        coords = node.findtext('./gml:pointProperty/gml:Point/gml:coordinates', namespaces=NS)
        if not node_id or not coords:
            continue
        parts = [p.strip() for p in coords.split(',')]
        if len(parts) < 2:
            continue
        try:
            nodes[node_id] = (float(parts[0]), float(parts[1]))
        except ValueError:
            continue
    return nodes


def _parse_edges(root: ET.Element, nodes: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[str, str]]:
    edges: Dict[str, Tuple[str, str]] = {}
    for edge in root.findall('.//rcr:edgelist/gml:Edge', NS):
        edge_id = edge.get(GML_ID)
        refs = edge.findall('./gml:directedNode', NS)
        if not edge_id or len(refs) != 2:
            continue
        a = (refs[0].get(XLINK_HREF) or '').lstrip('#')
        b = (refs[1].get(XLINK_HREF) or '').lstrip('#')
        if a in nodes and b in nodes:
            edges[edge_id] = (a, b)
    return edges


def _building_centroid(face: ET.Element, edges: Dict[str, Tuple[str, str]], nodes: Dict[str, Tuple[float, float]]) -> Tuple[float, float] | None:
    seen = []
    unique = set()
    for directed_edge in face.findall('./gml:directedEdge', NS):
        edge_ref = (directed_edge.get(XLINK_HREF) or '').lstrip('#')
        if edge_ref not in edges:
            continue
        node_a, node_b = edges[edge_ref]
        for node_id in (node_a, node_b):
            if node_id not in unique:
                unique.add(node_id)
                seen.append(nodes[node_id])
    if not seen:
        return None
    sx = sum(p[0] for p in seen)
    sy = sum(p[1] for p in seen)
    return (sx / len(seen), sy / len(seen))


def parse_map_gml(map_gml_path: Path) -> MapInfo:
    root = ET.parse(map_gml_path).getroot()
    nodes = _parse_nodes(root)
    edges = _parse_edges(root, nodes)
    buildings: List[BuildingInfo] = []

    for building in root.findall('.//rcr:buildinglist/rcr:building', NS):
        entity_id = building.get(GML_ID)
        face = building.find('./gml:Face', NS)
        if not entity_id or face is None:
            continue
        centroid = _building_centroid(face, edges, nodes)
        if centroid is None:
            continue
        try:
            buildings.append(BuildingInfo(entity_id=int(entity_id), centroid=centroid))
        except ValueError:
            continue

    return MapInfo(buildings=buildings)


def classify_map_size(map_info: MapInfo) -> str:
    count = len(map_info.buildings)
    if count <= 80:
        return 'small'
    if count <= 220:
        return 'medium'
    return 'large'


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
