from __future__ import annotations

import json
from pathlib import Path

import pytest

from BaseRescueAgent.module.util.runtime_settings import runtime_settings, setting, project_root


@pytest.fixture(autouse=True)
def clear_settings_cache():
    runtime_settings.cache_clear()
    yield
    runtime_settings.cache_clear()


@pytest.mark.unit
def test_project_root_points_to_base_rescue_agent_root():
    root = project_root()
    assert (root / "main.py").exists()
    assert (root / "config" / "runtime").exists()


@pytest.mark.unit
def test_runtime_settings_merge_expected_modules():
    data = runtime_settings()

    assert "search" in data
    assert "detector" in data
    assert "path" in data
    assert "logs" in data
    assert setting("search.mode") in {"heuristic", "hybrid", "pure_ml_test", "shadow"}
    assert setting("detector.mode") in {"heuristic", "hybrid", "pure_ml_test", "shadow"}
    assert setting("path.mode") in {"heuristic", "hybrid", "pure_ml_test", "shadow"}


@pytest.mark.integration
def test_configured_model_files_exist():
    root = project_root()
    for key in ["search.model_path", "detector.model_path", "path.model_path"]:
        configured = setting(key)
        assert configured, f"{key} must be configured"
        assert (root / configured).exists(), f"missing configured model: {configured}"


@pytest.mark.integration
def test_json_configs_are_valid_and_contain_no_unknown_top_level_sections():
    root = project_root()
    allowed = {"logs", "agent", "ml_v2", "system", "search", "detector", "path"}
    for path in sorted((root / "config" / "runtime").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert set(data).issubset(allowed), f"unexpected top-level key in {path.name}: {set(data) - allowed}"
