from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

try:
    from .search_v2_model import SEARCH_V2_CANDIDATE_FEATURES, save_metadata_json
except ImportError:
    from search_v2_model import SEARCH_V2_CANDIDATE_FEATURES, save_metadata_json


def load_rows(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def matrix_from_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row.get(feature, 0.0) or 0.0) for feature in SEARCH_V2_CANDIDATE_FEATURES] for row in rows], dtype=np.float64)
    y = np.asarray([int(row.get('label', 0) or 0) for row in rows], dtype=np.int32)
    return x, y


def sample_weights(rows: list[dict]) -> np.ndarray:
    counts = defaultdict(int)
    positives = defaultdict(int)
    for row in rows:
        counts[row['decision_id']] += 1
        positives[row['decision_id']] += int(row.get('label', 0) or 0)
    weights: list[float] = []
    for row in rows:
        decision_id = row['decision_id']
        total = max(counts[decision_id], 1)
        pos = max(positives[decision_id], 1)
        neg = max(total - pos, 1)
        label = int(row.get('label', 0) or 0)
        if label == 1:
            base = total / (2.0 * pos)
            outcome_multiplier = float(row.get('outcome_score', 1.0) or 1.0)
            weights.append(base * max(outcome_multiplier, 0.2))
        else:
            weights.append(total / (2.0 * neg))
    return np.asarray(weights, dtype=np.float64)


def decision_top1_accuracy(rows: list[dict], probabilities: np.ndarray) -> float:
    by_decision = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        by_decision[row['decision_id']].append((float(probability), int(row['label'])))
    if not by_decision:
        return 0.0
    correct = 0
    for items in by_decision.values():
        items.sort(key=lambda item: item[0], reverse=True)
        if items[0][1] == 1:
            correct += 1
    return correct / max(len(by_decision), 1)


def evaluate(rows: list[dict], probabilities: np.ndarray) -> dict:
    by_decision = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        by_decision[row['decision_id']].append((float(probability), int(row['label'])))
    mrr_total = 0.0
    for items in by_decision.values():
        items.sort(key=lambda item: item[0], reverse=True)
        reciprocal_rank = 0.0
        for index, (_, label) in enumerate(items, start=1):
            if label == 1:
                reciprocal_rank = 1.0 / index
                break
        mrr_total += reciprocal_rank
    positive_rows = [row for row in rows if int(row.get('label', 0) or 0) == 1]
    return {
        'rows': len(rows),
        'decisions': len(by_decision),
        'top1_accuracy': round(decision_top1_accuracy(rows, probabilities), 6),
        'mrr': round(mrr_total / max(len(by_decision), 1), 6),
        'positive_rate': round(float(np.mean([int(row.get('label', 0) or 0) for row in rows])) if rows else 0.0, 6),
        'positive_outcome_mean': round(float(np.mean([float(row.get('outcome_score', 1.0) or 1.0) for row in positive_rows])) if positive_rows else 0.0, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--group-field', default='run_id')
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_path)
    if not rows:
        raise ValueError('Dataset is empty')

    groups = np.asarray([row.get(args.group_field) or row.get('run_id') or row['decision_id'] for row in rows])
    indices = np.arange(len(rows))
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, valid_idx = next(splitter.split(indices, groups=groups))

    train_rows = [rows[index] for index in train_idx]
    valid_rows = [rows[index] for index in valid_idx]

    x_train, y_train = matrix_from_rows(train_rows)
    x_valid, _ = matrix_from_rows(valid_rows)
    w_train = sample_weights(train_rows)

    model = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=300, min_samples_leaf=24, l2_regularization=0.05, random_state=args.random_state)
    model.fit(x_train, y_train, sample_weight=w_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    valid_prob = model.predict_proba(x_valid)[:, 1]

    report = {
        'version': 'search_v2',
        'group_field': args.group_field,
        'train': evaluate(train_rows, train_prob),
        'valid': evaluate(valid_rows, valid_prob),
        'feature_count': len(SEARCH_V2_CANDIDATE_FEATURES),
        'features': SEARCH_V2_CANDIDATE_FEATURES,
        'train_groups': sorted({row.get(args.group_field) or row.get('run_id') for row in train_rows}),
        'valid_groups': sorted({row.get(args.group_field) or row.get('run_id') for row in valid_rows}),
    }

    x_all, y_all = matrix_from_rows(rows)
    w_all = sample_weights(rows)
    final_model = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=360, min_samples_leaf=24, l2_regularization=0.05, random_state=args.random_state)
    final_model.fit(x_all, y_all, sample_weight=w_all)

    artifact = {
        'model': final_model,
        'features': SEARCH_V2_CANDIDATE_FEATURES,
        'metadata': {'version': 'search_v2', 'group_field': args.group_field, 'report_path': str((output_dir / 'search_v2_report.json').resolve())},
    }
    joblib.dump(artifact, output_dir / 'search_v2.joblib')
    save_metadata_json(output_dir / 'search_v2_report.json', report)
    save_metadata_json(output_dir / 'search_v2_model_meta.json', artifact['metadata'])


if __name__ == '__main__':
    main()
