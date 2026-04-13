from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from BaseRescueAgent.ml.path_edge_risk_model import PATH_EDGE_RISK_V3_FEATURES, save_metadata_json


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'main.py').exists() and (parent / 'runtime').exists():
            return parent
    return Path.cwd()


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return default


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def matrix_from_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[_safe_float(row.get(feature)) for feature in PATH_EDGE_RISK_V3_FEATURES] for row in rows], dtype=np.float64)
    y = np.asarray([int(_safe_float(row.get('label'))) for row in rows], dtype=np.int32)
    return x, y


def sample_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = defaultdict(int)
    positives = defaultdict(int)
    for row in rows:
        counts[row['decision_id']] += 1
        positives[row['decision_id']] += int(_safe_float(row.get('label')))
    weights: list[float] = []
    for row in rows:
        decision_id = row['decision_id']
        total = max(counts[decision_id], 1)
        pos = max(positives[decision_id], 1)
        neg = max(total - pos, 1)
        label = int(_safe_float(row.get('label')))
        base = total / (2.0 * (pos if label == 1 else neg))
        custom = max(_safe_float(row.get('sample_weight'), 1.0), 0.1)
        weights.append(base * custom)
    return np.asarray(weights, dtype=np.float64)


def decision_top1_accuracy(rows: list[dict[str, Any]], probabilities: np.ndarray) -> float:
    by_decision = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        by_decision[row['decision_id']].append((float(probability), int(_safe_float(row.get('label')))))
    if not by_decision:
        return 0.0
    correct = 0
    for items in by_decision.values():
        items.sort(key=lambda item: item[0], reverse=True)
        if items[0][1] == 1:
            correct += 1
    return correct / max(len(by_decision), 1)


def evaluate(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    by_decision = defaultdict(list)
    label_sources = Counter()
    override_decisions = set()
    for row, probability in zip(rows, probabilities):
        decision_id = row['decision_id']
        label = int(_safe_float(row.get('label')))
        by_decision[decision_id].append((float(probability), label))
        label_sources[row.get('label_source') or 'unknown'] += 1
        if (row.get('label_source') or '') == 'shadow_override':
            override_decisions.add(decision_id)
    mrr_total = 0.0
    for items in by_decision.values():
        items.sort(key=lambda item: item[0], reverse=True)
        reciprocal_rank = 0.0
        for index, (_, label) in enumerate(items, start=1):
            if label == 1:
                reciprocal_rank = 1.0 / index
                break
        mrr_total += reciprocal_rank
    positive_rate = float(np.mean([int(_safe_float(row.get('label'))) for row in rows])) if rows else 0.0
    return {
        'rows': len(rows),
        'decisions': len(by_decision),
        'override_decisions': len(override_decisions),
        'top1_accuracy': round(decision_top1_accuracy(rows, probabilities), 6),
        'mrr': round(mrr_total / max(len(by_decision), 1), 6),
        'positive_rate': round(positive_rate, 6),
        'label_source_counts': dict(label_sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Train path edge risk v3 ranking model.')
    parser.add_argument('--dataset', type=str, default=str(project_root() / 'runtime' / 'datasets' / 'path_edge_risk_v3' / 'path_edge_risk_v3_dataset.csv'))
    parser.add_argument('--output-dir', type=str, default=str(project_root() / 'models' / 'path_edge_risk_v3'))
    parser.add_argument('--group-field', type=str, default='run_id')
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_path)
    if not rows:
        raise ValueError('Dataset is empty')
    positive_rows = sum(int(_safe_float(row.get('label'))) for row in rows)
    if positive_rows < 200:
        raise ValueError(f'Not enough positive rows for training: {positive_rows}')

    groups = np.asarray([row.get(args.group_field) or row.get('run_id') or row['decision_id'] for row in rows])
    indices = np.arange(len(rows))
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, valid_idx = next(splitter.split(indices, groups=groups))

    train_rows = [rows[index] for index in train_idx]
    valid_rows = [rows[index] for index in valid_idx]
    x_train, y_train = matrix_from_rows(train_rows)
    x_valid, _ = matrix_from_rows(valid_rows)
    w_train = sample_weights(train_rows)

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=360,
        min_samples_leaf=28,
        l2_regularization=0.08,
        random_state=args.random_state,
    )
    model.fit(x_train, y_train, sample_weight=w_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    valid_prob = model.predict_proba(x_valid)[:, 1]

    report = {
        'version': 'path_edge_risk_v3',
        'group_field': args.group_field,
        'train': evaluate(train_rows, train_prob),
        'valid': evaluate(valid_rows, valid_prob),
        'feature_count': len(PATH_EDGE_RISK_V3_FEATURES),
        'features': PATH_EDGE_RISK_V3_FEATURES,
        'train_groups': sorted({row.get(args.group_field) or row.get('run_id') for row in train_rows}),
        'valid_groups': sorted({row.get(args.group_field) or row.get('run_id') for row in valid_rows}),
    }

    x_all, y_all = matrix_from_rows(rows)
    w_all = sample_weights(rows)
    final_model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=420,
        min_samples_leaf=28,
        l2_regularization=0.08,
        random_state=args.random_state,
    )
    final_model.fit(x_all, y_all, sample_weight=w_all)

    artifact = {
        'model': final_model,
        'features': PATH_EDGE_RISK_V3_FEATURES,
        'metadata': {
            'version': 'path_edge_risk_v3',
            'group_field': args.group_field,
            'report_path': str((output_dir / 'path_edge_risk_v3_report.json').resolve()),
        },
    }
    joblib.dump(artifact, output_dir / 'path_edge_risk_v3.joblib')
    save_metadata_json(output_dir / 'path_edge_risk_v3_report.json', report)
    save_metadata_json(output_dir / 'path_edge_risk_v3_model_meta.json', artifact['metadata'])
    print(report)


if __name__ == '__main__':
    main()
