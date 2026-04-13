from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random, SystemRandom
from typing import List
import json
import shutil

from .gml_parser import classify_map_size, parse_map_gml
from .patterns import GenerationContext, PATTERNS
from .scenario_io import read_base_scenario, write_scenario


GENERATOR_VERSION = '3.0'


@dataclass(frozen=True)
class BaseMapPaths:
    name: str
    root_dir: Path
    map_dir: Path
    map_gml: Path
    scenario_xml: Path
    size_group: str
    building_count: int


@dataclass(frozen=True)
class GeneratedMapInfo:
    name: str
    output_dir: Path
    family: str
    seed: int
    base_map: str
    size_group: str


@dataclass(frozen=True)
class GenerationBatch:
    all_maps: List[GeneratedMapInfo]
    newly_generated: List[GeneratedMapInfo]
    request_seed: int


def discover_base_maps(maps_root: Path) -> List[BaseMapPaths]:
    result: List[BaseMapPaths] = []
    if not maps_root.exists():
        return result

    for child in sorted(maps_root.iterdir()):
        if not child.is_dir():
            continue
        map_dir = child / 'map'
        map_gml = map_dir / 'map.gml'
        scenario_xml = map_dir / 'scenario.xml'
        if not map_gml.exists() or not scenario_xml.exists():
            continue
        try:
            map_info = parse_map_gml(map_gml)
        except Exception:
            continue
        if not map_info.buildings:
            continue
        result.append(
            BaseMapPaths(
                name=child.name,
                root_dir=child,
                map_dir=map_dir,
                map_gml=map_gml,
                scenario_xml=scenario_xml,
                size_group=classify_map_size(map_info),
                building_count=len(map_info.buildings),
            )
        )
    return result


def _normalize_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return SystemRandom().randint(0, 10**9)


def allocate_generation_plan(base_maps: List[BaseMapPaths], total_maps: int, rng: Random) -> List[BaseMapPaths]:
    groups = {
        'small': [m for m in base_maps if m.size_group == 'small'],
        'medium': [m for m in base_maps if m.size_group == 'medium'],
        'large': [m for m in base_maps if m.size_group == 'large'],
    }
    ordered: List[BaseMapPaths] = []
    available_group_order = [name for name in ['small', 'medium', 'large'] if groups[name]]
    if not available_group_order:
        return ordered
    start_offset = rng.randrange(len(available_group_order))
    group_order = available_group_order[start_offset:] + available_group_order[:start_offset]
    indices = {k: 0 for k in groups}
    shuffled_groups = {}
    for name, items in groups.items():
        copied = items[:]
        rng.shuffle(copied)
        shuffled_groups[name] = copied

    while len(ordered) < total_maps:
        progress = False
        for group_name in group_order:
            items = shuffled_groups[group_name]
            if not items:
                continue
            if indices[group_name] > 0 and indices[group_name] % len(items) == 0:
                rng.shuffle(items)
            ordered.append(items[indices[group_name] % len(items)])
            indices[group_name] += 1
            progress = True
            if len(ordered) >= total_maps:
                break
        if not progress:
            break
    return ordered


def _build_family_plan(total_maps: int, rng: Random) -> List[str]:
    family_names = list(PATTERNS.keys())
    plan: List[str] = []
    while len(plan) < total_maps:
        batch = family_names[:]
        rng.shuffle(batch)
        plan.extend(batch)
    return plan[:total_maps]


def _copy_config(base_config_dir: Path, output_dir: Path) -> None:
    dst = output_dir / 'config'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base_config_dir, dst)


def _copy_map_gml(map_gml_src: Path, output_dir: Path) -> None:
    dst_dir = output_dir / 'map'
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(map_gml_src, dst_dir / 'map.gml')


def _meta_matches_request(meta: dict, civilian_count: int, request_seed: int) -> bool:
    return (
        str(meta.get('generator_version') or '') == GENERATOR_VERSION and
        int(meta.get('requested_base_civilian_count') or -1) == civilian_count and
        int(meta.get('requested_seed') or -1) == request_seed
    )


def _read_generated_info(path: Path, civilian_count: int, request_seed: int) -> GeneratedMapInfo | None:
    meta_path = path / 'meta.json'
    if not meta_path.exists() or not path.is_dir():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if not _meta_matches_request(meta, civilian_count, request_seed):
            return None
        return GeneratedMapInfo(
            name=str(meta.get('generated_name') or path.name),
            output_dir=path,
            family=str(meta.get('family') or ''),
            seed=int(meta.get('seed') or 0),
            base_map=str(meta.get('base_map') or ''),
            size_group=str(meta.get('size_group') or ''),
        )
    except Exception:
        return None


def collect_existing_generated_maps(output_root: Path, civilian_count: int, request_seed: int, cleanup_stale: bool = False) -> List[GeneratedMapInfo]:
    result: List[GeneratedMapInfo] = []
    if not output_root.exists():
        return result
    for child in sorted(output_root.iterdir()):
        info = _read_generated_info(child, civilian_count, request_seed)
        if info is not None:
            result.append(info)
            continue
        if cleanup_stale and child.is_dir() and (child / 'meta.json').exists():
            shutil.rmtree(child)
    result.sort(key=lambda item: item.name)
    return result


def generate_maps(
    base_config_dir: Path,
    maps_root: Path,
    output_root: Path,
    total_maps: int,
    civilian_count: int,
    seed: int | None,
) -> GenerationBatch:
    request_seed = _normalize_seed(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    existing = collect_existing_generated_maps(output_root, civilian_count, request_seed, cleanup_stale=True)
    if len(existing) >= total_maps:
        return GenerationBatch(all_maps=existing[:total_maps], newly_generated=[], request_seed=request_seed)

    base_maps = discover_base_maps(maps_root)
    if not base_maps:
        raise RuntimeError(f'Base maps not found in: {maps_root}')

    missing = total_maps - len(existing)
    master_rng = Random(request_seed)
    plan = allocate_generation_plan(base_maps, total_maps, master_rng)
    family_plan = _build_family_plan(total_maps, master_rng)
    newly_generated: List[GeneratedMapInfo] = []

    for index, base_map in enumerate(plan[len(existing): len(existing) + missing], start=len(existing)):
        map_info = parse_map_gml(base_map.map_gml)
        base_scenario = read_base_scenario(base_map.scenario_xml)
        refuge_ids = [r.location for r in base_scenario.refuges]
        candidate_default = base_scenario.ambulance_locations[0] if base_scenario.ambulance_locations else (refuge_ids[0] if refuge_ids else map_info.buildings[0].entity_id)
        family = family_plan[index]
        family_seed = master_rng.randint(0, 10**9)
        size_bonus = {'small': 0, 'medium': 2, 'large': 4}[base_map.size_group]
        effective_civilians = civilian_count + size_bonus

        ctx = GenerationContext(
            map_info=map_info,
            refuge_ids=refuge_ids,
            default_ambulance_location=candidate_default,
            civilian_count=effective_civilians,
        )
        pattern_result = PATTERNS[family](ctx, Random(family_seed))
        generated_name = f'{index + 1:03d}__{base_map.size_group}__{base_map.name}__{family}'
        out_dir = output_root / generated_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _copy_config(base_config_dir, out_dir)
        _copy_map_gml(base_map.map_gml, out_dir)
        write_scenario(out_dir / 'map' / 'scenario.xml', base_scenario.refuges, pattern_result.ambulance_location, pattern_result.civilians)

        meta = {
            'generated_name': generated_name,
            'base_map': base_map.name,
            'size_group': base_map.size_group,
            'family': family,
            'seed': family_seed,
            'requested_seed': request_seed,
            'requested_base_civilian_count': civilian_count,
            'generator_version': GENERATOR_VERSION,
            'effective_civilian_count': effective_civilians,
            'actual_civilian_count': len(pattern_result.civilians),
            'building_count': base_map.building_count,
            'description': pattern_result.description,
            'source_map_dir': str(base_map.root_dir),
            'source_config_dir': str(base_config_dir),
        }
        (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        newly_generated.append(GeneratedMapInfo(generated_name, out_dir, family, family_seed, base_map.name, base_map.size_group))

    all_maps = collect_existing_generated_maps(output_root, civilian_count, request_seed, cleanup_stale=False)[:total_maps]
    return GenerationBatch(all_maps=all_maps, newly_generated=newly_generated, request_seed=request_seed)
