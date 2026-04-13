from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .common_v2 import decision_key, iter_records, parse_allowed_modes, text_value, to_float, to_int, write_csv, write_jsonl
    from .detector_v3_model import DETECTOR_V3_FEATURES
except ImportError:
    from common_v2 import decision_key, iter_records, parse_allowed_modes, text_value, to_float, to_int, write_csv, write_jsonl
    from detector_v3_model import DETECTOR_V3_FEATURES


def phase_features(phase: str) -> dict[str, int]:
    phase = str(phase or '')
    return {
        'phase_search': 1 if phase == 'search' else 0,
        'phase_transport': 1 if phase == 'transport' else 0,
        'phase_move_to_victim': 1 if phase == 'move_to_victim' else 0,
    }


def compute_detector_outcome_score(outcome: dict[str, Any] | None) -> float:
    if not outcome:
        return 1.0
    score = 0.6
    carrying_now = bool(outcome.get('carrying_now'))
    refuge_target_set = bool(outcome.get('refuge_target_set'))
    target_still_active = bool(outcome.get('target_still_active'))
    target_changed = bool(outcome.get('target_changed'))
    due_reached = bool(outcome.get('due_reached'))
    elapsed_ticks = to_int(outcome.get('elapsed_ticks'), 0)
    resolved_reason = str(outcome.get('resolved_reason') or '')
    selection_life_margin = to_float(outcome.get('selection_life_margin'), 0.0)
    selection_total_trip = to_float(outcome.get('selection_total_trip_distance'), 0.0)

    if carrying_now:
        score += 2.8
        if elapsed_ticks <= 2:
            score += 0.9
        elif elapsed_ticks <= 4:
            score += 0.5
        elif elapsed_ticks >= 8:
            score -= 0.2
    if refuge_target_set:
        score += 1.4
    if target_still_active:
        score += 0.45
    if target_changed:
        score -= 0.8
    if due_reached and not carrying_now and not refuge_target_set:
        score -= 0.65
    if resolved_reason == 'superseded':
        score -= 0.45
    if selection_life_margin > 0.0 and carrying_now and selection_life_margin <= 40.0:
        score += 0.55
    if selection_life_margin > 0.0 and not carrying_now and selection_life_margin <= 20.0:
        score -= 0.35
    if selection_total_trip >= 45000.0 and not carrying_now and not refuge_target_set:
        score -= 0.2
    return max(score, 0.15)


def build_rows(input_path: Path, allowed_modes: set[str], min_candidate_count: int, min_decisions_per_run: int) -> list[dict[str, Any]]:
    decision_events: list[tuple[str, dict[str, Any]]] = []
    outcome_by_key: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}

    for source_name, record in iter_records(input_path, suffixes=('__detector.jsonl',)):
        event_type = record.get('event_type')
        if event_type == 'decision_snapshot':
            payload = record.get('payload') or {}
            if payload.get('decision_type') == 'detector':
                decision_events.append((source_name, record))
        elif event_type == 'detector_selection_outcome':
            payload = record.get('payload') or {}
            key = decision_key(
                text_value(record.get('run_id')),
                text_value(record.get('agent_id')),
                to_int(payload.get('selection_tick')),
                text_value(payload.get('selected_id')),
                'detector',
            )
            outcome_by_key[key] = payload

    by_run_decisions: dict[str, int] = defaultdict(int)
    pending: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for source_name, record in decision_events:
        payload = record.get('payload') or {}
        state = dict(payload.get('state') or {})
        mode = str(state.get('detector_mode') or payload.get('selection_mode') or 'heuristic')
        if mode not in allowed_modes:
            continue
        if payload.get('selected_reason') in {'no_candidates', 'carrying'}:
            continue
        candidates = list(payload.get('candidates') or [])
        if len(candidates) < min_candidate_count:
            continue
        run_id = text_value(record.get('run_id'))
        agent_id = text_value(record.get('agent_id'))
        tick = to_int(record.get('tick'))
        selected_id = text_value(payload.get('selected_id'))
        key = decision_key(run_id, agent_id, tick, selected_id, 'detector')
        outcome = outcome_by_key.get(key)
        by_run_decisions[run_id] += 1
        phase = text_value(state.get('phase') or '')
        context = {
            'source_name': source_name,
            'run_id': run_id,
            'agent_id': agent_id,
            'tick': tick,
            'decision_id': f"{run_id}|{agent_id}|{tick}|detector",
            'selected_id': selected_id,
            'selected_reason': text_value(payload.get('selected_reason')),
            'candidate_count': to_int(payload.get('candidate_count') or len(candidates)),
            'detector_mode': mode,
            'phase': phase,
            'known_refuges': to_int(state.get('known_refuges')),
            'known_civilians': to_int(state.get('known_civilians')),
            'cluster_candidate_count': to_int(state.get('cluster_candidate_count')),
            'global_candidate_count': to_int(state.get('global_candidate_count')),
            'scoped_candidate_count': to_int(state.get('scoped_candidate_count')),
            'candidate_scope': text_value(state.get('candidate_scope')),
            'scope_world_count': to_int(state.get('scope_world_count')),
            'scope_local_count': to_int(state.get('scope_local_count')),
            'deferred_rescue_active': to_int(state.get('deferred_rescue_active')),
            'selection_mode': text_value(payload.get('selection_mode') or mode),
            'selected_by': text_value(payload.get('selected_by') or 'heuristic'),
            'exploration_used': to_int(payload.get('exploration_used')),
            'selected_rank_by_heuristic': to_int(payload.get('selected_rank_by_heuristic'), -1),
            'selected_rank_by_ml': to_int(payload.get('selected_rank_by_ml'), -1),
            'selected_rank_by_final': to_int(payload.get('selected_rank_by_final'), -1),
            'top_k_size': len(list(payload.get('top_k_candidates') or [])),
            'outcome_available': 1 if outcome else 0,
            'outcome_score': compute_detector_outcome_score(outcome),
            'outcome_carrying_now': to_int(outcome.get('carrying_now')) if outcome else 0,
            'outcome_refuge_target_set': to_int(outcome.get('refuge_target_set')) if outcome else 0,
            'outcome_target_still_active': to_int(outcome.get('target_still_active')) if outcome else 0,
            'outcome_target_changed': to_int(outcome.get('target_changed')) if outcome else 0,
            'outcome_due_reached': to_int(outcome.get('due_reached')) if outcome else 0,
            'outcome_elapsed_ticks': to_int(outcome.get('elapsed_ticks')) if outcome else 0,
            'outcome_resolved_reason': text_value(outcome.get('resolved_reason')) if outcome else '',
            'selection_distance': to_float(outcome.get('selection_distance')) if outcome else 0.0,
            'selection_refuge_distance': to_float(outcome.get('selection_refuge_distance')) if outcome else 0.0,
            'selection_total_trip_distance': to_float(outcome.get('selection_total_trip_distance')) if outcome else 0.0,
            'selection_life_margin': to_float(outcome.get('selection_life_margin')) if outcome else 0.0,
            'selection_urgency': to_float(outcome.get('selection_urgency')) if outcome else 0.0,
            'selection_competitors': to_float(outcome.get('selection_competitors')) if outcome else 0.0,
            'selection_reachable': to_int(outcome.get('selection_reachable')) if outcome else 0,
            'selection_path_nodes': to_int(outcome.get('selection_path_nodes')) if outcome else 0,
        }
        context.update(phase_features(phase))
        for candidate in candidates:
            candidate_id = text_value(candidate.get('candidate_id'))
            dedup_key = (context['decision_id'], candidate_id)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            row = dict(context)
            row['candidate_id'] = candidate_id
            row['label'] = 1 if (bool(candidate.get('is_selected')) or (selected_id and candidate_id == selected_id)) else 0
            row['rank'] = to_int(candidate.get('rank'))
            row['heuristic_rank'] = to_int(candidate.get('heuristic_rank'), row['rank'])
            row['ml_rank'] = to_int(candidate.get('ml_rank'), -1)
            row['final_rank'] = to_int(candidate.get('final_rank'), row['rank'])
            for feature in DETECTOR_V3_FEATURES:
                if feature in row:
                    continue
                row[feature] = to_float(candidate.get(feature, state.get(feature)))
            pending.append(row)

    return [row for row in pending if by_run_decisions.get(row['run_id'], 0) >= min_decisions_per_run]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--allowed-detector-modes', default='heuristic,heuristic_fallback,hybrid')
    parser.add_argument('--min-candidate-count', type=int, default=2)
    parser.add_argument('--min-decisions-per-run', type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    allowed_modes = parse_allowed_modes(args.allowed_detector_modes, ('heuristic', 'heuristic_fallback', 'hybrid'))
    rows = build_rows(input_path, allowed_modes, args.min_candidate_count, args.min_decisions_per_run)
    write_jsonl(output_dir / 'detector_v3_dataset.jsonl', rows)
    if rows:
        write_csv(output_dir / 'detector_v3_dataset.csv', rows)
    summary = {
        'rows': len(rows),
        'decisions': len({row['decision_id'] for row in rows}),
        'runs': len({row['run_id'] for row in rows}),
        'positive_rows': sum(int(row['label']) for row in rows),
        'feature_count': len(DETECTOR_V3_FEATURES),
        'allowed_detector_modes': sorted(allowed_modes),
        'outcome_available_rows': sum(int(row['outcome_available']) for row in rows),
        'positive_outcome_mean': round(sum(float(row['outcome_score']) for row in rows if int(row['label']) == 1) / max(sum(1 for row in rows if int(row['label']) == 1), 1), 6),
    }
    with (output_dir / 'detector_v3_dataset_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
