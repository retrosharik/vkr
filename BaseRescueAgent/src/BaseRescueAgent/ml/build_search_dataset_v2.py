from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .common_v2 import decision_key, iter_records, parse_allowed_modes, text_value, to_float, to_int, write_csv, write_jsonl
    from .search_v2_model import SEARCH_V2_CANDIDATE_FEATURES
except ImportError:
    from common_v2 import decision_key, iter_records, parse_allowed_modes, text_value, to_float, to_int, write_csv, write_jsonl
    from search_v2_model import SEARCH_V2_CANDIDATE_FEATURES


def scope_features(scope: str) -> dict[str, int]:
    scope = str(scope or '')
    return {
        'scope_cluster_unvisited': 1 if scope == 'cluster_unvisited' else 0,
        'scope_outside_unvisited': 1 if scope == 'outside_unvisited' else 0,
        'scope_cluster_revisit': 1 if scope == 'cluster_revisit' else 0,
        'scope_outside_revisit': 1 if scope == 'outside_revisit' else 0,
    }


def phase_features(phase: str) -> dict[str, int]:
    phase = str(phase or '')
    return {
        'phase_search': 1 if phase == 'search' else 0,
        'phase_transport': 1 if phase == 'transport' else 0,
        'phase_move_to_victim': 1 if phase == 'move_to_victim' else 0,
    }


def compute_search_outcome_score(outcome: dict[str, Any] | None) -> float:
    if not outcome:
        return 1.0
    score = 1.0
    score += min(max(to_int(outcome.get('visited_count_gain')), 0), 4) * 0.5
    score += 1.5 if bool(outcome.get('target_visited')) else 0.0
    score += 1.0 if bool(outcome.get('detector_target_found')) else 0.0
    score += 0.25 if bool(outcome.get('due_reached')) else 0.0
    if str(outcome.get('resolved_reason') or '') == 'superseded':
        score -= 0.5
    return max(score, 0.2)


def build_rows(input_path: Path, allowed_modes: set[str], min_candidate_count: int, min_decisions_per_run: int) -> list[dict[str, Any]]:
    decision_events: list[tuple[str, dict[str, Any]]] = []
    outcome_by_key: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}

    for source_name, record in iter_records(input_path, suffixes=('__search.jsonl',)):
        event_type = record.get('event_type')
        if event_type == 'decision_snapshot':
            payload = record.get('payload') or {}
            if payload.get('decision_type') == 'search':
                decision_events.append((source_name, record))
        elif event_type == 'search_selection_outcome':
            payload = record.get('payload') or {}
            key = decision_key(
                text_value(record.get('run_id')),
                text_value(record.get('agent_id')),
                to_int(payload.get('selection_tick')),
                text_value(payload.get('selected_id')),
                'search',
            )
            outcome_by_key[key] = payload

    by_run_decisions: dict[str, int] = defaultdict(int)
    pending: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for source_name, record in decision_events:
        payload = record.get('payload') or {}
        state = dict(payload.get('state') or {})
        mode = str(state.get('search_mode') or payload.get('selection_mode') or 'heuristic')
        if mode not in allowed_modes:
            continue
        candidates = list(payload.get('candidates') or [])
        if len(candidates) < min_candidate_count:
            continue
        run_id = text_value(record.get('run_id'))
        agent_id = text_value(record.get('agent_id'))
        tick = to_int(record.get('tick'))
        selected_id = text_value(payload.get('selected_id'))
        key = decision_key(run_id, agent_id, tick, selected_id, 'search')
        outcome = outcome_by_key.get(key)
        by_run_decisions[run_id] += 1
        current_scope = text_value(state.get('search_scope') or '')
        phase = text_value(state.get('phase') or '')
        context = {
            'source_name': source_name,
            'run_id': run_id,
            'agent_id': agent_id,
            'tick': tick,
            'decision_id': f"{run_id}|{agent_id}|{tick}|search",
            'selected_id': selected_id,
            'selected_reason': text_value(payload.get('selected_reason')),
            'candidate_count': to_int(payload.get('candidate_count') or len(candidates)),
            'search_mode': mode,
            'phase': phase,
            'search_scope': current_scope,
            'known_civilians': to_int(state.get('known_civilians')),
            'known_refuges': to_int(state.get('known_refuges')),
            'cluster_candidate_count': to_int(state.get('cluster_candidate_count')),
            'global_candidate_count': to_int(state.get('global_candidate_count')),
            'cluster_unvisited_count': to_int(state.get('cluster_unvisited_count')),
            'outside_unvisited_count': to_int(state.get('outside_unvisited_count')),
            'cluster_revisit_count': to_int(state.get('cluster_revisit_count')),
            'outside_revisit_count': to_int(state.get('outside_revisit_count')),
            'cluster_remaining_ratio': to_float(state.get('cluster_remaining_ratio')),
            'forced_global': to_int(state.get('forced_global')),
            'selection_mode': text_value(payload.get('selection_mode') or mode),
            'selected_by': text_value(payload.get('selected_by') or 'heuristic'),
            'exploration_used': to_int(payload.get('exploration_used')),
            'selected_rank_by_heuristic': to_int(payload.get('selected_rank_by_heuristic'), -1),
            'selected_rank_by_ml': to_int(payload.get('selected_rank_by_ml'), -1),
            'selected_rank_by_final': to_int(payload.get('selected_rank_by_final'), -1),
            'top_k_size': len(list(payload.get('top_k_candidates') or [])),
            'outcome_available': 1 if outcome else 0,
            'outcome_score': compute_search_outcome_score(outcome),
            'outcome_visited_count_gain': to_int(outcome.get('visited_count_gain')) if outcome else 0,
            'outcome_target_visited': to_int(outcome.get('target_visited')) if outcome else 0,
            'outcome_detector_target_found': to_int(outcome.get('detector_target_found')) if outcome else 0,
            'outcome_due_reached': to_int(outcome.get('due_reached')) if outcome else 0,
            'outcome_elapsed_ticks': to_int(outcome.get('elapsed_ticks')) if outcome else 0,
            'outcome_resolved_reason': text_value(outcome.get('resolved_reason')) if outcome else '',
        }
        context.update(scope_features(current_scope))
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
            for feature in SEARCH_V2_CANDIDATE_FEATURES:
                if feature in row:
                    continue
                row[feature] = to_float(candidate.get(feature, state.get(feature)))
            pending.append(row)

    return [row for row in pending if by_run_decisions.get(row['run_id'], 0) >= min_decisions_per_run]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--allowed-search-modes', default='heuristic,heuristic_fallback')
    parser.add_argument('--min-candidate-count', type=int, default=3)
    parser.add_argument('--min-decisions-per-run', type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    allowed_modes = parse_allowed_modes(args.allowed_search_modes, ('heuristic', 'heuristic_fallback'))
    rows = build_rows(input_path, allowed_modes, args.min_candidate_count, args.min_decisions_per_run)
    write_jsonl(output_dir / 'search_v2_dataset.jsonl', rows)
    if rows:
        write_csv(output_dir / 'search_v2_dataset.csv', rows)
    summary = {
        'rows': len(rows),
        'decisions': len({row['decision_id'] for row in rows}),
        'runs': len({row['run_id'] for row in rows}),
        'positive_rows': sum(int(row['label']) for row in rows),
        'feature_count': len(SEARCH_V2_CANDIDATE_FEATURES),
        'allowed_search_modes': sorted(allowed_modes),
        'outcome_available_rows': sum(int(row['outcome_available']) for row in rows),
        'positive_outcome_mean': round(sum(float(row['outcome_score']) for row in rows if int(row['label']) == 1) / max(sum(1 for row in rows if int(row['label']) == 1), 1), 6),
    }
    with (output_dir / 'search_v2_dataset_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
