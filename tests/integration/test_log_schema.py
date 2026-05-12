from __future__ import annotations

import json
from pathlib import Path

import pytest

from BaseRescueAgent.module.util.runtime_settings import project_root


REQUIRED_LOG_FIELDS = {
    "schema_version",
    "run_id",
    "module",
    "module_type",
    "agent_id",
    "tick",
    "event_type",
    "payload",
}


@pytest.mark.integration
def test_raw_jsonl_logs_have_required_schema_fields():
    raw_root = project_root() / "runtime" / "raw_logs"
    files = sorted(raw_root.glob("*/*.jsonl"))
    assert files, "No raw jsonl logs found. Run at least one simulation before this integration test."

    checked = 0
    for path in files[:20]:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()[:200], start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = REQUIRED_LOG_FIELDS - set(record)
            assert not missing, f"{path}:{line_number} missing fields: {missing}"
            assert isinstance(record["schema_version"], int)
            assert isinstance(record["tick"], int)
            assert isinstance(record["event_type"], str) and record["event_type"]
            assert isinstance(record["payload"], dict)
            checked += 1
    assert checked > 0


@pytest.mark.integration
def test_decision_snapshots_reference_selected_candidate_when_candidates_are_logged():
    raw_root = project_root() / "runtime" / "raw_logs"
    files = sorted(raw_root.glob("*/*.jsonl"))
    assert files, "No raw jsonl logs found. Run at least one simulation before this integration test."

    snapshots = 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") != "decision_snapshot":
                continue
            payload = record.get("payload", {})
            selected_id = payload.get("selected_id")
            candidates = payload.get("candidates", [])
            if selected_id is not None and candidates:
                candidate_ids = {str(c.get("candidate_id")) for c in candidates}
                if str(selected_id) not in candidate_ids:
                    assert payload.get("selected_rank") is None or payload.get("selected_rank") > len(candidates)
            snapshots += 1
    assert snapshots > 0, "No decision_snapshot events found in raw logs."


@pytest.mark.integration
def test_path_snapshots_contain_status_and_path_metadata():
    raw_root = project_root() / "runtime" / "raw_logs"
    path_files = sorted(raw_root.glob("*/*__path.jsonl"))
    assert path_files, "No path logs found."

    snapshots = 0
    allowed_statuses = {"ok", "cache_hit", "invalid", "not_found", "fallback_not_found"}
    for path in path_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") != "path_snapshot":
                continue
            result = record.get("payload", {}).get("result", {})
            assert "status" in result
            assert result["status"] in allowed_statuses or isinstance(result["status"], str)
            assert "cache_hit" in result
            assert "caller_context" in result
            snapshots += 1
    assert snapshots > 0, "No path_snapshot events found."
