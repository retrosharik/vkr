from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.offline]

from BaseRescueAgent.ml.detector_v3_model import DetectorV3Model, save_metadata_json as save_detector_meta
from BaseRescueAgent.ml.path_edge_risk_model import PathEdgeRiskModel, save_metadata_json as save_path_meta
from BaseRescueAgent.ml.search_v2_model import SearchV2Model, save_metadata_json as save_search_meta


class ProbModel:
    def predict_proba(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        values = np.clip(0.1 + matrix.sum(axis=1) * 0.01, 0.05, 0.95)
        return np.column_stack([1.0 - values, values])


class MarginModel:
    def decision_function(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        return matrix.sum(axis=1) * 0.1


class PredictModel:
    def predict(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        return np.ones(matrix.shape[0]) * 0.75


def _dump_artifact(tmp_path: Path, name: str, model, *, features: list[str], version: str) -> Path:
    path = tmp_path / f"{name}.joblib"
    joblib.dump({"model": model, "features": features, "metadata": {"version": version}}, path)
    return path


def test_search_v2_model_scores_candidates_with_scope_and_phase_features(tmp_path: Path):
    model_path = _dump_artifact(tmp_path, "search", ProbModel(), features=["distance", "phase_search", "scope_cluster_unvisited"], version="search_v2_test")
    model = SearchV2Model(model_path)

    description = model.describe()
    assert description["version"] == "search_v2_test"
    assert description["feature_count"] == 3

    scores = model.score_candidates(
        {"phase": "search", "search_scope": "cluster_unvisited"},
        [{"candidate_id": "a", "distance": 10}, {"candidate_id": "b", "distance": 30}, {"distance": 99}],
    )
    assert set(scores) == {"a", "b"}
    assert scores["b"] > scores["a"]

    meta_path = tmp_path / "search_meta.json"
    save_search_meta(meta_path, {"ok": True})
    assert json.loads(meta_path.read_text(encoding="utf-8"))["ok"] is True


def test_detector_v3_model_uses_decision_function_branch(tmp_path: Path):
    model_path = _dump_artifact(tmp_path, "detector", MarginModel(), features=["distance", "phase_transport", "deferred_rescue_active"], version="detector_v3_test")
    model = DetectorV3Model(model_path)

    scores = model.score_candidates(
        {"phase": "transport", "deferred_rescue_active": True},
        [{"candidate_id": "c1", "distance": 10}, {"candidate_id": "c2", "distance": 20}],
    )
    assert set(scores) == {"c1", "c2"}
    assert 0.0 < scores["c1"] < 1.0
    assert scores["c2"] > scores["c1"]

    meta_path = tmp_path / "detector_meta.json"
    save_detector_meta(meta_path, {"version": "detector_v3_test"})
    assert "detector_v3_test" in meta_path.read_text(encoding="utf-8")


def test_path_edge_risk_model_supports_v1_v3_and_predict_fallback(tmp_path: Path):
    v1_path = _dump_artifact(tmp_path, "path_v1", ProbModel(), features=["path_distance", "phase_search", "caller_search"], version="path_edge_risk_v1")
    v1 = PathEdgeRiskModel(v1_path)
    risk = v1.score_path({"phase": "search", "caller_context": "search"}, {"path_distance": 100})
    assert risk is not None and 0.0 <= risk <= 1.0

    v3_path = _dump_artifact(tmp_path, "path_v3", ProbModel(), features=["candidate_risk", "candidate_final_cost", "phase_search", "caller_search"], version="path_edge_risk_v3")
    v3 = PathEdgeRiskModel(v3_path)
    quality = v3.score_path({"phase": "search", "caller_context": "search"}, {"candidate_risk": 0.2, "candidate_final_cost": 100, "baseline_final_cost": 100})
    assert quality is not None and 0.0 <= quality <= 1.0

    predict_path = _dump_artifact(tmp_path, "path_predict", PredictModel(), features=["path_distance"], version="path_edge_risk_v1")
    predict_model = PathEdgeRiskModel(predict_path)
    assert predict_model.score_path({}, {"path_distance": 1}) == pytest.approx(0.75)

    meta_path = tmp_path / "path_meta.json"
    save_path_meta(meta_path, {"kind": "path"})
    assert json.loads(meta_path.read_text(encoding="utf-8"))["kind"] == "path"


def test_model_wrappers_accept_plain_joblib_model_without_artifact_dict(tmp_path: Path):
    path = tmp_path / "plain.joblib"
    joblib.dump(PredictModel(), path)
    model = SearchV2Model(path)
    assert model.metadata == {}
    assert model.features
    scores = model.score_candidates({}, [{"candidate_id": "plain"}])
    assert scores["plain"] == pytest.approx(0.75)
