from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.offline]

from BaseRescueAgent.ml import train_detector_v3 as train_detector
from BaseRescueAgent.ml import train_path_edge_risk_v3 as train_path
from BaseRescueAgent.ml import train_search_v2 as train_search
from BaseRescueAgent.ml.detector_v3_model import DETECTOR_V3_FEATURES
from BaseRescueAgent.ml.path_edge_risk_model import PATH_EDGE_RISK_V3_FEATURES
from BaseRescueAgent.ml.search_v2_model import SEARCH_V2_CANDIDATE_FEATURES


class DummyClassifier:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def fit(self, x, y, sample_weight=None):
        self.n_features_ = x.shape[1]
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            p = np.zeros(x.shape[0])
        else:
            p = np.clip(0.35 + x[:, 0] * 0.001, 0.05, 0.95)
        return np.column_stack([1.0 - p, p])


def _row(decision_id: str, run_id: str, label: int, features: list[str], **extra) -> dict:
    row = {"decision_id": decision_id, "run_id": run_id, "label": str(label)}
    for index, feature in enumerate(features):
        row[feature] = str((index + 1) * (1 if label else 0.5))
    row.update({key: str(value) for key, value in extra.items()})
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_train_search_helpers_compute_matrix_weights_and_metrics(tmp_path: Path):
    rows = [
        _row("d1", "r1", 1, SEARCH_V2_CANDIDATE_FEATURES, outcome_score=2.0),
        _row("d1", "r1", 0, SEARCH_V2_CANDIDATE_FEATURES, outcome_score=1.0),
        _row("d2", "r2", 0, SEARCH_V2_CANDIDATE_FEATURES),
        _row("d2", "r2", 1, SEARCH_V2_CANDIDATE_FEATURES, outcome_score=0.1),
    ]
    csv_path = tmp_path / "search.csv"
    _write_csv(csv_path, rows)

    loaded = train_search.load_rows(csv_path)
    x, y = train_search.matrix_from_rows(loaded)
    assert x.shape == (4, len(SEARCH_V2_CANDIDATE_FEATURES))
    assert y.tolist() == [1, 0, 0, 1]
    weights = train_search.sample_weights(loaded)
    assert weights[0] > weights[1]
    assert train_search.decision_top1_accuracy(loaded, np.asarray([0.9, 0.1, 0.2, 0.8])) == 1.0
    assert train_search.decision_top1_accuracy([], np.asarray([])) == 0.0
    report = train_search.evaluate(loaded, np.asarray([0.9, 0.1, 0.2, 0.8]))
    assert report["decisions"] == 2
    assert report["mrr"] == 1.0


def test_train_detector_helpers_apply_rank_sensitive_weights(tmp_path: Path):
    rows = [
        _row("d1", "r1", 1, DETECTOR_V3_FEATURES, outcome_score=2.0, selected_rank_by_heuristic=1, heuristic_rank=1),
        _row("d1", "r1", 0, DETECTOR_V3_FEATURES, outcome_score=1.0, selected_rank_by_heuristic=2, heuristic_rank=2),
        _row("d2", "r2", 0, DETECTOR_V3_FEATURES, heuristic_rank=5),
        _row("d2", "r2", 1, DETECTOR_V3_FEATURES, outcome_score=1.5, selected_rank_by_heuristic=3, heuristic_rank=3),
    ]
    csv_path = tmp_path / "detector.csv"
    _write_csv(csv_path, rows)

    loaded = train_detector.load_rows(csv_path)
    x, y = train_detector.matrix_from_rows(loaded)
    assert x.shape == (4, len(DETECTOR_V3_FEATURES))
    assert y.sum() == 2
    weights = train_detector.sample_weights(loaded)
    assert weights[0] > weights[2]
    assert train_detector.decision_top1_accuracy(loaded, np.asarray([0.8, 0.7, 0.1, 0.9])) == 1.0
    report = train_detector.evaluate(loaded, np.asarray([0.8, 0.7, 0.1, 0.9]))
    assert report["positive_rate"] == 0.5


def test_train_path_helpers_include_label_sources_and_custom_weights(tmp_path: Path):
    rows = [
        _row("d1", "r1", 1, PATH_EDGE_RISK_V3_FEATURES, sample_weight=20.0, label_source="shadow_override"),
        _row("d1", "r1", 0, PATH_EDGE_RISK_V3_FEATURES, sample_weight=10.0, label_source="shadow_override"),
        _row("d2", "r2", 0, PATH_EDGE_RISK_V3_FEATURES, sample_weight=0.01, label_source="baseline_keep_downsampled"),
        _row("d2", "r2", 1, PATH_EDGE_RISK_V3_FEATURES, sample_weight=2.0, label_source="safe_challenger"),
    ]
    csv_path = tmp_path / "path.csv"
    _write_csv(csv_path, rows)

    loaded = train_path.load_rows(csv_path)
    x, y = train_path.matrix_from_rows(loaded)
    assert x.shape == (4, len(PATH_EDGE_RISK_V3_FEATURES))
    assert y.tolist() == [1, 0, 0, 1]
    assert train_path._safe_float("bad", 3.0) == 3.0
    weights = train_path.sample_weights(loaded)
    assert weights[0] > weights[2]
    assert train_path.decision_top1_accuracy(loaded, np.asarray([0.9, 0.1, 0.2, 0.8])) == 1.0
    report = train_path.evaluate(loaded, np.asarray([0.9, 0.1, 0.2, 0.8]))
    assert report["override_decisions"] == 1
    assert report["label_source_counts"]["shadow_override"] == 2


def test_training_mains_write_artifacts_with_dummy_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(train_search, "HistGradientBoostingClassifier", DummyClassifier)
    monkeypatch.setattr(train_detector, "HistGradientBoostingClassifier", DummyClassifier)
    monkeypatch.setattr(train_path, "HistGradientBoostingClassifier", DummyClassifier)

    search_rows = []
    detector_rows = []
    path_rows = []
    for group in range(4):
        for decision in range(2):
            decision_id = f"d{group}-{decision}"
            run_id = f"r{group}"
            search_rows.append(_row(decision_id, run_id, 1, SEARCH_V2_CANDIDATE_FEATURES, outcome_score=1.5))
            search_rows.append(_row(decision_id, run_id, 0, SEARCH_V2_CANDIDATE_FEATURES, outcome_score=1.0))
            detector_rows.append(_row(decision_id, run_id, 1, DETECTOR_V3_FEATURES, outcome_score=1.4, selected_rank_by_heuristic=1, heuristic_rank=1))
            detector_rows.append(_row(decision_id, run_id, 0, DETECTOR_V3_FEATURES, outcome_score=1.0, selected_rank_by_heuristic=2, heuristic_rank=2))
            for idx in range(30):
                path_rows.append(_row(f"{decision_id}-{idx}", run_id, 1, PATH_EDGE_RISK_V3_FEATURES, sample_weight=2.0, label_source="safe_challenger"))
                path_rows.append(_row(f"{decision_id}-{idx}", run_id, 0, PATH_EDGE_RISK_V3_FEATURES, sample_weight=1.0, label_source="baseline_keep_downsampled"))

    search_csv = tmp_path / "search.csv"
    detector_csv = tmp_path / "detector.csv"
    path_csv = tmp_path / "path.csv"
    _write_csv(search_csv, search_rows)
    _write_csv(detector_csv, detector_rows)
    _write_csv(path_csv, path_rows)

    search_out = tmp_path / "search_model"
    detector_out = tmp_path / "detector_model"
    path_out = tmp_path / "path_model"

    monkeypatch.setattr(sys, "argv", ["train_search_v2", "--dataset", str(search_csv), "--output-dir", str(search_out), "--test-size", "0.25"])
    train_search.main()
    monkeypatch.setattr(sys, "argv", ["train_detector_v3", "--dataset", str(detector_csv), "--output-dir", str(detector_out), "--test-size", "0.25"])
    train_detector.main()
    monkeypatch.setattr(sys, "argv", ["train_path_edge_risk_v3", "--dataset", str(path_csv), "--output-dir", str(path_out), "--test-size", "0.25"])
    train_path.main()

    assert joblib.load(search_out / "search_v2.joblib")["metadata"]["version"] == "search_v2"
    assert joblib.load(detector_out / "detector_v3.joblib")["metadata"]["version"] == "detector_v3"
    assert joblib.load(path_out / "path_edge_risk_v3.joblib")["metadata"]["version"] == "path_edge_risk_v3"
    assert (search_out / "search_v2_report.json").exists()
    assert (detector_out / "detector_v3_report.json").exists()
    assert (path_out / "path_edge_risk_v3_report.json").exists()


def test_training_mains_reject_empty_or_too_small_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    empty = tmp_path / "empty.csv"
    _write_csv(empty, [])
    monkeypatch.setattr(sys, "argv", ["train_search_v2", "--dataset", str(empty), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="Dataset is empty"):
        train_search.main()

    small_path_csv = tmp_path / "small_path.csv"
    _write_csv(small_path_csv, [_row("d", "r", 1, PATH_EDGE_RISK_V3_FEATURES, sample_weight=1.0, label_source="safe_challenger")])
    monkeypatch.setattr(sys, "argv", ["train_path_edge_risk_v3", "--dataset", str(small_path_csv), "--output-dir", str(tmp_path / "path_out")])
    with pytest.raises(ValueError, match="Not enough positive rows"):
        train_path.main()
