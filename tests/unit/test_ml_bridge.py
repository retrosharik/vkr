from __future__ import annotations

import pytest

from BaseRescueAgent.module.util import ml_bridge
from BaseRescueAgent.module.util.ml_bridge import MlBridge
from BaseRescueAgent.module.util.runtime_settings import runtime_settings


@pytest.fixture(autouse=True)
def reset_ml_cache():
    ml_bridge._MODEL_CACHE.clear()
    ml_bridge._MODEL_ERRORS.clear()
    runtime_settings.cache_clear()
    yield
    ml_bridge._MODEL_CACHE.clear()
    ml_bridge._MODEL_ERRORS.clear()
    runtime_settings.cache_clear()


class DummyModel:
    def __init__(self, path):
        self.path = path

    def describe(self):
        return {"dummy": True, "path": str(self.path)}

    def score_candidates(self, context, candidates):
        return {item["candidate_id"]: index / 10 for index, item in enumerate(candidates, start=1)}

    def score_path(self, context, path_payload):
        return 0.25


@pytest.mark.unit
def test_ml_bridge_uses_heuristic_when_ml_flag_is_disabled():
    bridge = MlBridge("path")

    assert bridge.is_requested() is False
    assert bridge.is_active() is False
    assert bridge.mode_name() == "heuristic"
    assert bridge.score_candidates({}, [{"candidate_id": "1"}]) == {}
    assert bridge.score_path({}, {}) is None


@pytest.mark.unit
def test_ml_bridge_falls_back_when_model_class_is_unavailable(monkeypatch):
    monkeypatch.setattr(ml_bridge, "SearchV2Model", None)

    bridge = MlBridge("search")

    assert bridge.is_requested() is True
    assert bridge.is_active() is False
    assert bridge.mode_name() == "heuristic_fallback"
    assert bridge.score_candidates({}, [{"candidate_id": "1"}]) == {}


@pytest.mark.unit
def test_ml_bridge_scores_candidates_with_loaded_model(monkeypatch):
    monkeypatch.setattr(ml_bridge, "SearchV2Model", DummyModel)

    bridge = MlBridge("search")
    scores = bridge.score_candidates({}, [{"candidate_id": "a"}, {"candidate_id": "b"}])

    assert bridge.is_active() is True
    assert bridge.mode_name() == "hybrid"
    assert scores == {"a": 0.1, "b": 0.2}
    assert bridge.describe()["model"]["dummy"] is True


@pytest.mark.unit
def test_ml_bridge_score_path_returns_float_for_path_model(monkeypatch):
    monkeypatch.setattr(ml_bridge, "PathEdgeRiskModel", DummyModel)
    bridge = MlBridge("path")
    bridge.requested_mode = "shadow"

    assert bridge.score_path({}, {"path": [1, 2, 3]}) == 0.25
