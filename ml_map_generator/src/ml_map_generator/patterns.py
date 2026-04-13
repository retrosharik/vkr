from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List

from .gml_parser import BuildingInfo, MapInfo, euclidean


@dataclass(frozen=True)
class PatternResult:
    civilians: List[int]
    ambulance_location: int
    description: str


@dataclass(frozen=True)
class GenerationContext:
    map_info: MapInfo
    refuge_ids: List[int]
    default_ambulance_location: int
    civilian_count: int


def _building_dict(buildings: List[BuildingInfo]) -> Dict[int, BuildingInfo]:
    return {b.entity_id: b for b in buildings}


def _building_ids_excluding_refuges(ctx: GenerationContext) -> List[int]:
    refuge_set = set(ctx.refuge_ids)
    ids = [b.entity_id for b in ctx.map_info.buildings if b.entity_id not in refuge_set]
    if not ids:
        ids = [b.entity_id for b in ctx.map_info.buildings]
    return ids


def _nearest_to(center_id: int, candidates: List[int], building_map: Dict[int, BuildingInfo]) -> List[int]:
    center = building_map[center_id].centroid
    return sorted(candidates, key=lambda x: euclidean(center, building_map[x].centroid))


def _farthest_from(center_id: int, candidates: List[int], building_map: Dict[int, BuildingInfo]) -> List[int]:
    center = building_map[center_id].centroid
    return sorted(candidates, key=lambda x: euclidean(center, building_map[x].centroid), reverse=True)


def _sample_with_replacement(rng: Random, ids: List[int], count: int) -> List[int]:
    if not ids:
        return []
    return [rng.choice(ids) for _ in range(count)]


def _sample_without_replacement_or_repeat(rng: Random, ids: List[int], count: int) -> List[int]:
    if not ids:
        return []
    if count <= len(ids):
        return rng.sample(ids, count)
    result = ids[:]
    while len(result) < count:
        result.append(rng.choice(ids))
    rng.shuffle(result)
    return result[:count]


def _is_large(ctx: GenerationContext) -> bool:
    return len(ctx.map_info.buildings) > 220


def _spawn_near_primary_cluster(ctx: GenerationContext, cluster_candidates: List[int], fallback: int, building_map: Dict[int, BuildingInfo]) -> int:
    if not _is_large(ctx):
        return fallback
    pool = [cid for cid in cluster_candidates if cid in building_map]
    if not pool:
        return fallback
    anchor = pool[0]
    neighborhood = _nearest_to(anchor, _building_ids_excluding_refuges(ctx), building_map)[:20]
    if not neighborhood:
        return fallback
    return neighborhood[min(2, len(neighborhood) - 1)]


def _inject_local_seed_group(ctx: GenerationContext, civilians: List[int], spawn: int, building_map: Dict[int, BuildingInfo], count: int) -> List[int]:
    if not _is_large(ctx) or spawn not in building_map:
        return civilians
    candidate_ids = _building_ids_excluding_refuges(ctx)
    local = [bid for bid in _nearest_to(spawn, candidate_ids, building_map) if bid != spawn][:8]
    if not local:
        return civilians
    seed_group = local[:max(2, min(4, count))]
    seeded = seed_group[:] + civilians
    return seeded[:max(len(civilians), count)]


def _spread_out_candidates(candidate_ids: List[int], building_map: Dict[int, BuildingInfo], count: int, rng: Random) -> List[int]:
    if not candidate_ids or count <= 0:
        return []
    if count >= len(candidate_ids):
        shuffled = candidate_ids[:]
        rng.shuffle(shuffled)
        return shuffled
    selected = [rng.choice(candidate_ids)]
    remaining = [bid for bid in candidate_ids if bid != selected[0]]
    while remaining and len(selected) < count:
        next_id = max(
            remaining,
            key=lambda bid: min(
                euclidean(building_map[bid].centroid, building_map[sid].centroid)
                for sid in selected
            ),
        )
        selected.append(next_id)
        remaining.remove(next_id)
    rng.shuffle(selected)
    return selected[:count]


def _random_partition(rng: Random, total: int, buckets: int, minimum_each: int = 1) -> List[int]:
    if buckets <= 0:
        return []
    if total <= 0:
        return [0 for _ in range(buckets)]
    values = [minimum_each for _ in range(buckets)]
    remaining = max(0, total - minimum_each * buckets)
    for _ in range(remaining):
        values[rng.randrange(buckets)] += 1
    rng.shuffle(values)
    return values


def clustered_near_refuge(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    refuge_anchor = rng.choice(ctx.refuge_ids) if ctx.refuge_ids else rng.choice(candidate_ids)
    close_span = max(3, min(len(candidate_ids), rng.randint(4, 10)))
    close = _nearest_to(refuge_anchor, candidate_ids, bm)[:close_span]
    civilians = _sample_with_replacement(rng, close, ctx.civilian_count)
    spawn = _spawn_near_primary_cluster(ctx, close, refuge_anchor, bm)
    civilians = _inject_local_seed_group(ctx, civilians, spawn, bm, ctx.civilian_count)
    return PatternResult(civilians, spawn, 'cluster near refuge')


def clustered_far_from_refuge(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    refuge_anchor = rng.choice(ctx.refuge_ids) if ctx.refuge_ids else rng.choice(candidate_ids)
    far_span = max(4, min(len(candidate_ids), rng.randint(6, 14)))
    far = _farthest_from(refuge_anchor, candidate_ids, bm)[:far_span]
    cluster_center = rng.choice(far)
    cluster_members = _nearest_to(cluster_center, far, bm)[: max(3, min(len(far), rng.randint(3, 7)))]
    civilians = _sample_with_replacement(rng, cluster_members, ctx.civilian_count)
    spawn = _spawn_near_primary_cluster(ctx, cluster_members, ctx.default_ambulance_location, bm)
    civilians = _inject_local_seed_group(ctx, civilians, spawn, bm, ctx.civilian_count)
    return PatternResult(civilians, spawn, 'far cluster from refuge')


def sparse_uniform(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    civilians = _sample_without_replacement_or_repeat(rng, candidate_ids, ctx.civilian_count)
    spawn = _spawn_near_primary_cluster(ctx, civilians[:6], ctx.default_ambulance_location, bm)
    civilians = _inject_local_seed_group(ctx, civilians, spawn, bm, ctx.civilian_count)
    return PatternResult(civilians, spawn, 'sparse uniform distribution')


def one_big_cluster_plus_outliers(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    anchor = rng.choice(candidate_ids)
    cluster = _nearest_to(anchor, candidate_ids, bm)[: max(3, min(len(candidate_ids), rng.randint(4, 8)))]
    rest = [x for x in candidate_ids if x not in cluster]
    main_ratio = rng.uniform(0.6, 0.8)
    main_count = max(1, int(ctx.civilian_count * main_ratio))
    outlier_count = max(0, ctx.civilian_count - main_count)
    civilians = _sample_with_replacement(rng, cluster, main_count)
    civilians += _sample_without_replacement_or_repeat(rng, rest or candidate_ids, outlier_count)
    rng.shuffle(civilians)
    spawn = _spawn_near_primary_cluster(ctx, cluster, ctx.default_ambulance_location, bm)
    civilians = _inject_local_seed_group(ctx, civilians, spawn, bm, ctx.civilian_count)
    return PatternResult(civilians, spawn, 'one big cluster with outliers')


def deceptive_local_cluster(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    spawn = ctx.default_ambulance_location if ctx.default_ambulance_location in bm else rng.choice(candidate_ids)
    near = _nearest_to(spawn, candidate_ids, bm)[: max(3, min(len(candidate_ids), rng.randint(4, 7)))]
    far = _farthest_from(spawn, candidate_ids, bm)[: max(5, min(len(candidate_ids), rng.randint(8, 14)))]
    local_count = max(1, int(ctx.civilian_count * rng.uniform(0.2, 0.35)))
    remote_count = max(1, ctx.civilian_count - local_count)
    civilians = _sample_with_replacement(rng, near, local_count)
    civilians += _sample_with_replacement(rng, far, remote_count)
    rng.shuffle(civilians)
    civilians = _inject_local_seed_group(ctx, civilians, spawn, bm, ctx.civilian_count)
    return PatternResult(civilians, spawn, 'small local lure + better remote cluster')


def many_equal_candidates(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    anchor = rng.choice(candidate_ids)
    band = _nearest_to(anchor, candidate_ids, bm)[: max(8, min(len(candidate_ids), rng.randint(10, 16)))]
    civilians = _sample_without_replacement_or_repeat(rng, band, ctx.civilian_count)
    spawn = _spawn_near_primary_cluster(ctx, band, ctx.default_ambulance_location, bm)
    civilians = _inject_local_seed_group(ctx, civilians, spawn, bm, ctx.civilian_count)
    return PatternResult(civilians, spawn, 'many near-equal candidates')


def hard_far_edge_case(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    spawn = ctx.default_ambulance_location if ctx.default_ambulance_location in bm else rng.choice(candidate_ids)
    far = _farthest_from(spawn, candidate_ids, bm)
    edge_targets = far[: max(2, min(len(far), rng.randint(3, 6)))]
    civilians = _sample_with_replacement(rng, edge_targets, ctx.civilian_count)
    spawn = _spawn_near_primary_cluster(ctx, edge_targets, spawn, bm)
    civilians = _inject_local_seed_group(ctx, civilians, spawn, bm, ctx.civilian_count)
    return PatternResult(civilians, spawn, 'hard far edge case')


def dispersed_small_groups(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    group_centers = _sample_without_replacement_or_repeat(rng, candidate_ids, max(3, min(len(candidate_ids), ctx.civilian_count // 2 or 1)))
    civilians: List[int] = []
    for center in group_centers:
        group = _nearest_to(center, candidate_ids, bm)[:3]
        group_size = rng.randint(1, min(3, len(group)))
        civilians.extend(group[:group_size])
        if len(civilians) >= ctx.civilian_count:
            break
    if len(civilians) < ctx.civilian_count:
        civilians.extend(_sample_without_replacement_or_repeat(rng, candidate_ids, ctx.civilian_count - len(civilians)))
    civilians = civilians[:ctx.civilian_count]
    spawn = _spawn_near_primary_cluster(ctx, civilians[:6], ctx.default_ambulance_location, bm)
    civilians = _inject_local_seed_group(ctx, civilians, spawn, bm, ctx.civilian_count)
    return PatternResult(civilians, spawn, 'dispersed small victim groups')


def widely_dispersed_singletons(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    spread_targets = _spread_out_candidates(candidate_ids, bm, min(ctx.civilian_count, len(candidate_ids)), rng)
    civilians = spread_targets[:]
    if len(civilians) < ctx.civilian_count:
        rest = [bid for bid in candidate_ids if bid not in civilians]
        civilians.extend(_sample_without_replacement_or_repeat(rng, rest or candidate_ids, ctx.civilian_count - len(civilians)))
    spawn = ctx.default_ambulance_location if ctx.default_ambulance_location in bm else rng.choice(candidate_ids)
    return PatternResult(civilians[:ctx.civilian_count], spawn, 'widely dispersed single civilians over the map')


def dense_start_cluster(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    spawn = ctx.default_ambulance_location if ctx.default_ambulance_location in bm else rng.choice(candidate_ids)
    near_candidates = [bid for bid in _nearest_to(spawn, candidate_ids, bm) if bid != spawn]
    if not near_candidates:
        near_candidates = candidate_ids[:]
    building_count = max(3, min(len(near_candidates), rng.randint(3, 5)))
    hotspot_buildings = near_candidates[:building_count]
    total_victims = max(ctx.civilian_count, rng.randint(15, 20))
    distribution = _random_partition(rng, total_victims, len(hotspot_buildings), minimum_each=1)
    civilians: List[int] = []
    for building_id, count in zip(hotspot_buildings, distribution):
        civilians.extend([building_id] * count)
    rng.shuffle(civilians)
    return PatternResult(civilians, spawn, 'dense casualty hotspot near the starting ambulance location')


def two_remote_clusters(ctx: GenerationContext, rng: Random) -> PatternResult:
    bm = _building_dict(ctx.map_info.buildings)
    candidate_ids = _building_ids_excluding_refuges(ctx)
    spawn = ctx.default_ambulance_location if ctx.default_ambulance_location in bm else rng.choice(candidate_ids)
    far_from_spawn = _farthest_from(spawn, candidate_ids, bm)
    first_anchor = far_from_spawn[0] if far_from_spawn else rng.choice(candidate_ids)
    second_anchor_pool = [bid for bid in _farthest_from(first_anchor, candidate_ids, bm) if bid != first_anchor]
    second_anchor = second_anchor_pool[0] if second_anchor_pool else first_anchor
    cluster_a = _nearest_to(first_anchor, candidate_ids, bm)[: max(3, min(len(candidate_ids), rng.randint(3, 6)))]
    cluster_b = _nearest_to(second_anchor, candidate_ids, bm)[: max(3, min(len(candidate_ids), rng.randint(3, 6)))]
    total = ctx.civilian_count
    first_count = max(1, total // 2)
    second_count = max(1, total - first_count)
    civilians = _sample_with_replacement(rng, cluster_a, first_count)
    civilians += _sample_with_replacement(rng, cluster_b, second_count)
    rng.shuffle(civilians)
    return PatternResult(civilians[:total], spawn, 'two remote clusters on opposite sides of the map')


PATTERNS = {
    'clustered_near_refuge': clustered_near_refuge,
    'clustered_far_from_refuge': clustered_far_from_refuge,
    'sparse_uniform': sparse_uniform,
    'one_big_cluster_plus_outliers': one_big_cluster_plus_outliers,
    'deceptive_local_cluster': deceptive_local_cluster,
    'many_equal_candidates': many_equal_candidates,
    'hard_far_edge_case': hard_far_edge_case,
    'dispersed_small_groups': dispersed_small_groups,
    'widely_dispersed_singletons': widely_dispersed_singletons,
    'dense_start_cluster': dense_start_cluster,
    'two_remote_clusters': two_remote_clusters,
}
