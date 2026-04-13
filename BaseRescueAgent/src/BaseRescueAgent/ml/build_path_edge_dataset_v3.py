from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from BaseRescueAgent.ml.common_v2 import iter_records, to_float, to_int, text_value, write_csv, write_jsonl
from BaseRescueAgent.ml.path_edge_risk_model import PATH_EDGE_RISK_V3_FEATURES, save_metadata_json

SHADOW_BUILD_TAGS = {'path_shadow_stage2_v1', 'path_shadow_stage3_v1'}


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'main.py').exists() and (parent / 'runtime').exists():
            return parent
    return Path.cwd()


def _phase_context_features(row: dict[str, Any], request: dict[str, Any]) -> None:
    phase = text_value(request.get('phase'))
    caller_context = text_value(request.get('caller_context'))
    row['phase_search'] = 1.0 if phase == 'search' else 0.0
    row['phase_transport'] = 1.0 if phase == 'transport' else 0.0
    row['phase_move_to_victim'] = 1.0 if phase == 'move_to_victim' else 0.0
    row['caller_action_move'] = 1.0 if caller_context == 'action_move' else 0.0
    row['caller_action_transport'] = 1.0 if caller_context == 'action_transport' else 0.0
    row['caller_action_clear'] = 1.0 if caller_context == 'action_clear' else 0.0
    row['caller_command_executor'] = 1.0 if caller_context == 'command_executor' else 0.0
    row['caller_search'] = 1.0 if caller_context == 'search' else 0.0
    row['caller_detector'] = 1.0 if caller_context == 'detector' else 0.0


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _safe_challenger(candidate: dict[str, Any], baseline_risk: float) -> bool:
    candidate_risk = to_float(candidate.get('risk'))
    final_cost_ratio = to_float(candidate.get('final_cost'), 1e18) / max(to_float(candidate.get('baseline_final_cost', 1.0), 1.0), 1.0)
    risk_improvement = baseline_risk - candidate_risk
    return (
        risk_improvement >= 0.08
        and final_cost_ratio <= 1.06
        and not bool(candidate.get('backtracks'))
        and not bool(candidate.get('blocked_history'))
        and to_float(candidate.get('fail_count')) <= 1.0
    )


def _row_from_candidate(record: dict[str, Any], request: dict[str, Any], result: dict[str, Any], ml_info: dict[str, Any],
                        decision_id: str, candidate: dict[str, Any], label: int, label_source: str, sample_weight: float,
                        baseline_first_hop: str, ml_best_first_hop: str, baseline_summary: dict[str, Any],
                        baseline_risk: float, baseline_base_cost: float, baseline_final_cost: float, baseline_distance: float,
                        baseline_risk_high: float, candidate_count: int, rank_by_final: dict[str, int], rank_by_risk: dict[str, int],
                        override_reason: str) -> dict[str, Any]:
    first_hop = text_value(candidate.get('first_hop'))
    candidate_risk = to_float(candidate.get('risk'))
    candidate_base_cost = to_float(candidate.get('base_cost'))
    candidate_final_cost = to_float(candidate.get('final_cost'))
    candidate_distance = to_float(candidate.get('distance'))
    row: dict[str, Any] = {
        'run_id': text_value(record.get('run_id')),
        'agent_id': text_value(record.get('agent_id')),
        'tick': to_int(record.get('tick')),
        'decision_id': decision_id,
        'from': text_value(request.get('from')),
        'to': text_value(request.get('to')),
        'first_hop': first_hop,
        'label': int(label),
        'label_source': label_source,
        'sample_weight': float(sample_weight),
        'would_override_if_enabled': 1.0 if bool(ml_info.get('would_override_if_enabled')) else 0.0,
        'would_override_reason_if_enabled': override_reason,
        'candidate_risk': candidate_risk,
        'candidate_base_cost': candidate_base_cost,
        'candidate_final_cost': candidate_final_cost,
        'candidate_distance': candidate_distance,
        'candidate_fail_count': to_float(candidate.get('fail_count')),
        'candidate_backtracks': 1.0 if bool(candidate.get('backtracks')) else 0.0,
        'candidate_blocked_history': 1.0 if bool(candidate.get('blocked_history')) else 0.0,
        'candidate_count': float(candidate_count),
        'candidate_rank_by_final_cost': float(rank_by_final.get(first_hop, candidate_count + 1)),
        'candidate_rank_by_risk': float(rank_by_risk.get(first_hop, candidate_count + 1)),
        'baseline_risk': baseline_risk,
        'baseline_base_cost': baseline_base_cost,
        'baseline_final_cost': baseline_final_cost,
        'baseline_risk_high': baseline_risk_high,
        'is_baseline_first_hop': 1.0 if first_hop == baseline_first_hop else 0.0,
        'is_ml_best_first_hop': 1.0 if first_hop == ml_best_first_hop else 0.0,
        'risk_improvement_vs_baseline': baseline_risk - candidate_risk,
        'base_cost_delta_vs_baseline': candidate_base_cost - baseline_base_cost,
        'base_cost_ratio_vs_baseline': (candidate_base_cost / baseline_base_cost) if baseline_base_cost > 0 else 1.0,
        'final_cost_delta_vs_baseline': candidate_final_cost - baseline_final_cost,
        'final_cost_ratio_vs_baseline': (candidate_final_cost / baseline_final_cost) if baseline_final_cost > 0 else 1.0,
        'distance_delta_vs_baseline': candidate_distance - baseline_distance,
        'distance_ratio_vs_baseline': (candidate_distance / baseline_distance) if baseline_distance > 0 else 1.0,
        'stationary_ticks': to_float(request.get('stationary_ticks')),
        'startup_recovery_locked': 1.0 if bool(request.get('startup_recovery_locked')) else 0.0,
        'blocked_first_hops_active_count': float(len(result.get('blocked_first_hops_active') or [])),
        'skipped_start_edges_count': float(len(result.get('skipped_start_edges') or [])),
        'path_node_count': to_float(result.get('node_count')),
        'path_distance': to_float(result.get('distance')),
        'path_expanded': to_float(result.get('expanded')),
        'path_logic_version': text_value(request.get('path_logic_version') or ml_info.get('path_logic_version')),
        'build_tag': text_value(record.get('build_tag')),
        'caller_context': text_value(request.get('caller_context')),
        'phase': text_value(request.get('phase')),
    }
    _phase_context_features(row, request)
    return row


def _candidate_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    payload = record.get('payload') or {}
    request = payload.get('request') or {}
    result = payload.get('result') or {}
    ml_info = result.get('ml_first_hop_info') or {}
    if not isinstance(ml_info, dict):
        return []
    mode = text_value(request.get('mode'))
    build_tag = text_value(record.get('build_tag'))
    path_logic_version = text_value(request.get('path_logic_version') or ml_info.get('path_logic_version'))
    if mode != 'shadow':
        return []
    if not (build_tag in SHADOW_BUILD_TAGS or path_logic_version in SHADOW_BUILD_TAGS):
        return []

    candidate_summaries = ml_info.get('candidate_summaries') or []
    if not isinstance(candidate_summaries, list) or len(candidate_summaries) < 2:
        return []

    baseline_first_hop = text_value(ml_info.get('baseline_first_hop'))
    ml_best_first_hop = text_value(ml_info.get('ml_best_first_hop'))
    baseline_summary = ml_info.get('baseline_summary') or {}
    baseline_risk = to_float(ml_info.get('baseline_risk'))
    baseline_base_cost = to_float(ml_info.get('baseline_base_cost'))
    baseline_final_cost = to_float(ml_info.get('baseline_final_cost'))
    baseline_distance = to_float(baseline_summary.get('distance') or result.get('distance'))
    baseline_risk_high = 1.0 if bool(ml_info.get('baseline_risk_high')) else 0.0
    override_reason = text_value(ml_info.get('would_override_reason_if_enabled') or ml_info.get('would_override_reason'))

    sorted_by_final = sorted(candidate_summaries, key=lambda item: (to_float(item.get('final_cost'), 1e18), text_value(item.get('first_hop'))))
    sorted_by_risk = sorted(candidate_summaries, key=lambda item: (to_float(item.get('risk'), 1e18), text_value(item.get('first_hop'))))
    rank_by_final = {text_value(item.get('first_hop')): index + 1 for index, item in enumerate(sorted_by_final)}
    rank_by_risk = {text_value(item.get('first_hop')): index + 1 for index, item in enumerate(sorted_by_risk)}

    decision_id = '::'.join([
        text_value(record.get('run_id')),
        text_value(record.get('agent_id')),
        str(to_int(record.get('tick'))),
        text_value(request.get('from')),
        text_value(request.get('to')),
        text_value(request.get('caller_context')),
    ])
    decision_key = f"{decision_id}::{baseline_first_hop}::{ml_best_first_hop}"

    candidate_count = len(candidate_summaries)
    candidates_by_hop = {text_value(item.get('first_hop')): item for item in candidate_summaries}
    baseline_candidate = candidates_by_hop.get(baseline_first_hop)
    if baseline_candidate is None:
        return []

    rows: list[dict[str, Any]] = []
    shadow_override = bool(ml_info.get('would_override_if_enabled')) and ml_best_first_hop and ml_best_first_hop != baseline_first_hop and ml_best_first_hop in candidates_by_hop

    if shadow_override:
        pos_hop = ml_best_first_hop
        for candidate in candidate_summaries:
            first_hop = text_value(candidate.get('first_hop'))
            label = 1 if first_hop == pos_hop else 0
            sample_weight = 20.0 if label == 1 else (10.0 if first_hop == baseline_first_hop else 4.0)
            rows.append(_row_from_candidate(record, request, result, ml_info, decision_id, candidate, label, 'shadow_override', sample_weight,
                                            baseline_first_hop, ml_best_first_hop, baseline_summary, baseline_risk, baseline_base_cost,
                                            baseline_final_cost, baseline_distance, baseline_risk_high, candidate_count, rank_by_final,
                                            rank_by_risk, override_reason))
        return rows

    safe_candidates = [c for c in candidate_summaries if text_value(c.get('first_hop')) != baseline_first_hop and _safe_challenger(c, baseline_risk)]
    if safe_candidates:
        safe_candidates.sort(key=lambda item: (to_float(item.get('risk')), to_float(item.get('final_cost')), text_value(item.get('first_hop'))))
        pos_hop = text_value(safe_candidates[0].get('first_hop'))
        keep_hops = {baseline_first_hop, pos_hop}
        for candidate in candidate_summaries:
            first_hop = text_value(candidate.get('first_hop'))
            if first_hop not in keep_hops and rank_by_risk.get(first_hop, 99) > 2 and rank_by_final.get(first_hop, 99) > 2:
                continue
            label = 1 if first_hop == pos_hop else 0
            sample_weight = 12.0 if label == 1 else (8.0 if first_hop == baseline_first_hop else 3.0)
            rows.append(_row_from_candidate(record, request, result, ml_info, decision_id, candidate, label, 'safe_challenger', sample_weight,
                                            baseline_first_hop, ml_best_first_hop, baseline_summary, baseline_risk, baseline_base_cost,
                                            baseline_final_cost, baseline_distance, baseline_risk_high, candidate_count, rank_by_final,
                                            rank_by_risk, override_reason))
        return rows

    if _stable_fraction(decision_key) > 0.18:
        return []
    keep_candidates = []
    for candidate in candidate_summaries:
        first_hop = text_value(candidate.get('first_hop'))
        if first_hop == baseline_first_hop:
            keep_candidates.append(candidate)
            continue
        if rank_by_risk.get(first_hop, 99) <= 2 or rank_by_final.get(first_hop, 99) <= 2:
            keep_candidates.append(candidate)
    dedup = {text_value(item.get('first_hop')): item for item in keep_candidates}
    for candidate in dedup.values():
        first_hop = text_value(candidate.get('first_hop'))
        label = 1 if first_hop == baseline_first_hop else 0
        sample_weight = 2.0 if label == 1 else 1.0
        rows.append(_row_from_candidate(record, request, result, ml_info, decision_id, candidate, label, 'baseline_keep_downsampled', sample_weight,
                                        baseline_first_hop, ml_best_first_hop, baseline_summary, baseline_risk, baseline_base_cost,
                                        baseline_final_cost, baseline_distance, baseline_risk_high, candidate_count, rank_by_final,
                                        rank_by_risk, override_reason))
    return rows


def build_dataset(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decision_ids = set()
    runs = set()
    label_counts = Counter()
    label_source_counts = Counter()
    override_decisions = 0
    for _, record in iter_records(input_path, suffixes=('__path.jsonl',)):
        if record.get('event_type') != 'path_snapshot':
            continue
        candidate_rows = _candidate_rows(record)
        if not candidate_rows:
            continue
        rows.extend(candidate_rows)
        decision_id = candidate_rows[0]['decision_id']
        decision_ids.add(decision_id)
        runs.add(candidate_rows[0]['run_id'])
        if any(row['label_source'] == 'shadow_override' for row in candidate_rows):
            override_decisions += 1
        for row in candidate_rows:
            label_counts[int(row['label'])] += 1
            label_source_counts[row['label_source']] += 1
    summary = {
        'rows': len(rows),
        'decisions': len(decision_ids),
        'runs': len(runs),
        'positive_rows': int(label_counts.get(1, 0)),
        'negative_rows': int(label_counts.get(0, 0)),
        'override_decisions': int(override_decisions),
        'label_source_counts': dict(label_source_counts),
        'feature_count': len(PATH_EDGE_RISK_V3_FEATURES),
        'features': list(PATH_EDGE_RISK_V3_FEATURES),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Build path edge risk v3 dataset from shadow path logs.')
    parser.add_argument('--runtime-raw-logs', type=str, default=str(project_root() / 'runtime' / 'raw_logs'))
    parser.add_argument('--output-dir', type=str, default=str(project_root() / 'runtime' / 'datasets' / 'path_edge_risk_v3'))
    args = parser.parse_args()

    input_path = Path(args.runtime_raw_logs).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = build_dataset(input_path)
    if not rows:
        raise ValueError('No path v3 rows were generated from the provided logs')
    write_jsonl(output_dir / 'path_edge_risk_v3_dataset.jsonl', rows)
    write_csv(output_dir / 'path_edge_risk_v3_dataset.csv', rows)
    save_metadata_json(output_dir / 'path_edge_risk_v3_dataset_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
