from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.offline]

from BaseRescueAgent.ml import build_detector_dataset_v3 as detector_builder
from BaseRescueAgent.ml import build_path_edge_dataset_v3 as path_builder
from BaseRescueAgent.ml import build_search_dataset_v2 as search_builder


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _search_decision(tick: int, selected_id: str = "10") -> dict:
    return {
        "event_type": "decision_snapshot",
        "run_id": "run-a",
        "agent_id": "1",
        "tick": tick,
        "payload": {
            "decision_type": "search",
            "selected_id": selected_id,
            "selected_reason": "best_candidate",
            "candidate_count": 2,
            "selection_mode": "heuristic",
            "state": {
                "search_mode": "heuristic",
                "search_scope": "cluster_unvisited",
                "phase": "search",
                "known_civilians": 3,
                "known_refuges": 1,
                "cluster_candidate_count": 2,
                "global_candidate_count": 5,
                "cluster_unvisited_count": 2,
                "outside_unvisited_count": 3,
                "cluster_remaining_ratio": 0.4,
                "forced_global": 0,
            },
            "candidates": [
                {"candidate_id": selected_id, "is_selected": True, "rank": 1, "distance": 100.0, "centrality": 0.8, "reachable": 1},
                {"candidate_id": "11", "is_selected": False, "rank": 2, "distance": 300.0, "centrality": 0.1, "reachable": 1},
            ],
        },
    }


def test_search_dataset_builder_extracts_labeled_rows_and_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_dir = tmp_path / "logs"
    records = [
        _search_decision(1),
        {
            "event_type": "search_selection_outcome",
            "run_id": "run-a",
            "agent_id": "1",
            "tick": 3,
            "payload": {
                "selection_tick": 1,
                "selected_id": "10",
                "visited_count_gain": 2,
                "target_visited": True,
                "detector_target_found": True,
                "due_reached": True,
            },
        },
        _search_decision(2, selected_id="12"),
    ]
    _write_jsonl(log_dir / "run__search.jsonl", records)

    rows = search_builder.build_rows(log_dir, {"heuristic"}, min_candidate_count=2, min_decisions_per_run=1)
    assert len(rows) == 4
    selected = [row for row in rows if row["label"] == 1]
    assert selected[0]["scope_cluster_unvisited"] == 1
    assert selected[0]["phase_search"] == 1
    assert selected[0]["outcome_available"] == 1
    assert selected[0]["outcome_score"] > 1.0
    assert search_builder.compute_search_outcome_score({"resolved_reason": "superseded"}) < 1.0
    assert search_builder.scope_features("outside_revisit")["scope_outside_revisit"] == 1
    assert search_builder.phase_features("transport")["phase_transport"] == 1

    output_dir = tmp_path / "dataset"
    monkeypatch.setattr(sys, "argv", ["build_search_dataset_v2", "--input", str(log_dir), "--output-dir", str(output_dir), "--min-candidate-count", "2", "--min-decisions-per-run", "1"])
    search_builder.main()
    assert (output_dir / "search_v2_dataset.jsonl").exists()
    assert (output_dir / "search_v2_dataset.csv").exists()
    summary = json.loads((output_dir / "search_v2_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 4
    assert summary["positive_rows"] == 2


def _detector_decision(tick: int, selected_id: str = "200") -> dict:
    return {
        "event_type": "decision_snapshot",
        "run_id": "run-d",
        "agent_id": "7",
        "tick": tick,
        "payload": {
            "decision_type": "detector",
            "selected_id": selected_id,
            "selected_reason": "best_candidate",
            "candidate_count": 2,
            "selection_mode": "hybrid",
            "state": {
                "detector_mode": "hybrid",
                "phase": "move_to_victim",
                "known_civilians": 4,
                "known_refuges": 1,
                "cluster_candidate_count": 2,
                "global_candidate_count": 4,
                "scoped_candidate_count": 2,
                "deferred_rescue_active": 1,
            },
            "candidates": [
                {"candidate_id": selected_id, "is_selected": True, "rank": 1, "heuristic_rank": 1, "distance": 50, "hp": 9000, "damage": 25, "buriedness": 2, "reachable": 1},
                {"candidate_id": "201", "is_selected": False, "rank": 2, "heuristic_rank": 2, "distance": 400, "hp": 7000, "damage": 10, "buriedness": 0, "reachable": 1},
            ],
        },
    }


def test_detector_dataset_builder_extracts_outcomes_and_skips_invalid_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_dir = tmp_path / "logs"
    records = [
        _detector_decision(10),
        {
            "event_type": "detector_selection_outcome",
            "run_id": "run-d",
            "agent_id": "7",
            "tick": 12,
            "payload": {
                "selection_tick": 10,
                "selected_id": "200",
                "carrying_now": True,
                "refuge_target_set": True,
                "target_still_active": True,
                "elapsed_ticks": 2,
                "selection_life_margin": 35,
                "selection_total_trip_distance": 1200,
                "selection_reachable": 1,
            },
        },
        _detector_decision(11, selected_id="202"),
        {"event_type": "decision_snapshot", "run_id": "run-d", "agent_id": "7", "tick": 12, "payload": {"decision_type": "detector", "selected_reason": "carrying", "candidates": []}},
    ]
    _write_jsonl(log_dir / "run__detector.jsonl", records)

    rows = detector_builder.build_rows(log_dir, {"hybrid"}, min_candidate_count=2, min_decisions_per_run=1)
    assert len(rows) == 4
    assert [row for row in rows if row["label"] == 1][0]["phase_move_to_victim"] == 1
    assert [row for row in rows if row["label"] == 1][0]["outcome_carrying_now"] == 1
    assert detector_builder.compute_detector_outcome_score({"carrying_now": True, "elapsed_ticks": 2}) > 3.0
    assert detector_builder.compute_detector_outcome_score({"due_reached": True, "target_changed": True}) < 0.6
    assert detector_builder.phase_features("search")["phase_search"] == 1

    output_dir = tmp_path / "dataset"
    monkeypatch.setattr(sys, "argv", ["build_detector_dataset_v3", "--input", str(log_dir), "--output-dir", str(output_dir), "--min-candidate-count", "2", "--min-decisions-per-run", "1"])
    detector_builder.main()
    assert (output_dir / "detector_v3_dataset.jsonl").exists()
    assert (output_dir / "detector_v3_dataset.csv").exists()
    summary = json.loads((output_dir / "detector_v3_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 4
    assert summary["positive_rows"] == 2


def _path_shadow_record(*, override: bool = True, mode: str = "shadow") -> dict:
    return {
        "event_type": "path_snapshot",
        "run_id": "run-p",
        "agent_id": "3",
        "tick": 20,
        "build_tag": "path_shadow_stage2_v1",
        "payload": {
            "request": {
                "mode": mode,
                "from": "1",
                "to": "9",
                "caller_context": "search",
                "phase": "search",
                "stationary_ticks": 2,
                "startup_recovery_locked": False,
                "path_logic_version": "path_shadow_stage3_v1",
            },
            "result": {
                "node_count": 3,
                "distance": 1500,
                "expanded": 9,
                "blocked_first_hops_active": ["1->2"],
                "skipped_start_edges": [],
                "ml_first_hop_info": {
                    "baseline_first_hop": "2",
                    "ml_best_first_hop": "4",
                    "would_override_if_enabled": override,
                    "would_override_reason_if_enabled": "risk_improvement",
                    "baseline_risk": 0.55,
                    "baseline_base_cost": 1000,
                    "baseline_final_cost": 1000,
                    "baseline_risk_high": True,
                    "baseline_summary": {"first_hop": "2", "distance": 1000},
                    "candidate_summaries": [
                        {"first_hop": "2", "risk": 0.55, "base_cost": 1000, "final_cost": 1000, "distance": 1000, "fail_count": 0, "baseline_final_cost": 1000},
                        {"first_hop": "4", "risk": 0.20, "base_cost": 1020, "final_cost": 1020, "distance": 1020, "fail_count": 0, "baseline_final_cost": 1000},
                        {"first_hop": "5", "risk": 0.40, "base_cost": 1300, "final_cost": 1300, "distance": 1300, "fail_count": 2, "baseline_final_cost": 1000},
                    ],
                },
            },
        },
    }


def test_path_edge_dataset_builder_handles_shadow_override_and_main_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_dir = tmp_path / "logs"
    _write_jsonl(log_dir / "run__path.jsonl", [_path_shadow_record(override=True), _path_shadow_record(mode="heuristic")])

    rows, summary = path_builder.build_dataset(log_dir)
    assert len(rows) == 3
    assert summary["rows"] == 3
    assert summary["positive_rows"] == 1
    positive = [row for row in rows if row["label"] == 1][0]
    assert positive["first_hop"] == "4"
    assert positive["label_source"] == "shadow_override"
    assert positive["phase_search"] == 1.0
    assert positive["caller_search"] == 1.0
    assert positive["risk_improvement_vs_baseline"] > 0
    assert path_builder._safe_challenger({"risk": 0.2, "final_cost": 1020, "baseline_final_cost": 1000, "fail_count": 0}, 0.55)
    assert 0.0 <= path_builder._stable_fraction("stable-key") <= 1.0

    output_dir = tmp_path / "dataset"
    monkeypatch.setattr(sys, "argv", ["build_path_edge_dataset_v3", "--runtime-raw-logs", str(log_dir), "--output-dir", str(output_dir)])
    path_builder.main()
    assert (output_dir / "path_edge_risk_v3_dataset.jsonl").exists()
    assert (output_dir / "path_edge_risk_v3_dataset.csv").exists()
    summary_json = json.loads((output_dir / "path_edge_risk_v3_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary_json["override_decisions"] == 1


def test_path_edge_dataset_builder_safe_challenger_branch(tmp_path: Path):
    log_dir = tmp_path / "logs"
    _write_jsonl(log_dir / "run__path.jsonl", [_path_shadow_record(override=False)])
    rows, summary = path_builder.build_dataset(log_dir)
    assert rows
    assert summary["label_source_counts"]["safe_challenger"] >= 1
    assert {row["label"] for row in rows} == {0, 1}


def test_path_edge_main_rejects_empty_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_dir = tmp_path / "dataset"
    monkeypatch.setattr(sys, "argv", ["build_path_edge_dataset_v3", "--runtime-raw-logs", str(log_dir), "--output-dir", str(output_dir)])
    with pytest.raises(ValueError, match="No path v3 rows"):
        path_builder.main()
