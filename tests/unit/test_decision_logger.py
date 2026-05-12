from __future__ import annotations

import json

import pytest
from rcrscore.entities import EntityID

from BaseRescueAgent.module.util import decision_logger
from BaseRescueAgent.module.util.decision_logger import DecisionLogger
from tests.conftest import FakeAgentInfo, FakeWorldInfo


def eid(value: int) -> EntityID:
    return EntityID(value)


@pytest.fixture
def isolated_logger(monkeypatch, tmp_path):
    monkeypatch.setenv("RRS_BENCHMARK_RUN_ID", "unit_run")
    monkeypatch.setattr(decision_logger, "project_root", lambda: tmp_path)

    defaults = {
        "logs.raw_dir": "runtime/raw_logs",
        "logs.my_dir": "runtime/my_logs",
        "logs.debug_dir": "runtime/log",
        "logs.schema_version": 2,
        "logs.max_logged_candidates": 2,
        "logs.build_tag": "unit",
        "system.build_tag": "unit-build",
    }

    monkeypatch.setattr(decision_logger, "setting", lambda path, default=None: defaults.get(path, default))
    return tmp_path


@pytest.mark.unit
def test_decision_logger_writes_raw_debug_and_text_logs(isolated_logger):
    agent = FakeAgentInfo(agent_id=7, position=1, tick=12)
    logger = DecisionLogger(agent, FakeWorldInfo(), "UnitModule", "search")

    logger.log_text("hello", {"b": 2.34567, "a": "x"})
    logger.log_raw(None, "custom_event", {"value": 42})
    logger.debug_text("debug message", {"ratio": 0.123456})
    logger.debug_event("debug_event", {"ok": True})

    raw_records = [json.loads(line) for line in logger.raw_path.read_text(encoding="utf-8").splitlines()]
    debug_records = [json.loads(line) for line in logger.debug_json_path.read_text(encoding="utf-8").splitlines()]
    text_body = logger.text_path.read_text(encoding="utf-8")
    debug_text_body = logger.debug_text_path.read_text(encoding="utf-8")

    assert raw_records[-1]["event_type"] == "custom_event"
    assert raw_records[-1]["agent_id"] == "7"
    assert raw_records[-1]["tick"] == 12
    assert raw_records[-1]["build_tag"] == "unit-build"
    assert debug_records[-1]["event_type"] == "debug_event"
    assert "hello" in text_body and "b=2.3457" in text_body
    assert "debug message" in debug_text_body and "ratio=0.1235" in debug_text_body


@pytest.mark.unit
def test_decision_snapshot_trims_candidates_and_records_selected_rank(isolated_logger):
    logger = DecisionLogger(FakeAgentInfo(agent_id=7, position=1, tick=3), FakeWorldInfo(), "UnitModule", "detector")

    logger.decision_snapshot(
        "detector",
        state={"phase": "move_to_victim"},
        candidates=[
            {"candidate_id": "100", "heuristic_rank": 1, "ml_rank": 2, "final_rank": 1},
            {"candidate_id": "101", "heuristic_rank": 2, "ml_rank": 1, "final_rank": 2},
            {"candidate_id": "102", "heuristic_rank": 3, "ml_rank": 3, "final_rank": 3},
        ],
        selected_id="101",
        selected_reason="unit",
        metadata={"selection_mode": "hybrid", "selected_by": "ml_override", "top_k_candidates": ["100", "101"]},
    )

    records = [json.loads(line) for line in logger.raw_path.read_text(encoding="utf-8").splitlines()]
    payload = records[-1]["payload"]

    assert records[-1]["event_type"] == "decision_snapshot"
    assert payload["candidate_count"] == 3
    assert payload["logged_candidate_count"] == 2
    assert payload["selected_rank"] == 2
    assert payload["selected_rank_by_ml"] == 1
    assert payload["candidates"][1]["is_selected"] is True


@pytest.mark.unit
def test_path_and_state_snapshots_are_written_to_raw_log(isolated_logger):
    logger = DecisionLogger(FakeAgentInfo(agent_id=7, position=1, tick=5), FakeWorldInfo(), "UnitModule", "path")

    logger.path_snapshot({"from": "1", "to": "2"}, {"status": "ok", "cache_hit": False})
    logger.state_snapshot("state_snapshot", {"phase": "search"})

    records = [json.loads(line) for line in logger.raw_path.read_text(encoding="utf-8").splitlines()]

    assert records[-2]["event_type"] == "path_snapshot"
    assert records[-2]["payload"]["result"]["status"] == "ok"
    assert records[-1]["event_type"] == "state_snapshot"
    assert records[-1]["payload"]["phase"] == "search"
