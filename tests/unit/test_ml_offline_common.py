from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.offline]

from BaseRescueAgent.ml import common_v2 as common


def test_common_jsonl_directory_and_zip_iterators_roundtrip(tmp_path: Path):
    logs = tmp_path / "logs"
    nested = logs / "run"
    nested.mkdir(parents=True)
    (nested / "a__search.jsonl").write_text('{"a": 1}\n\n{"b": true}\n', encoding="utf-8")
    (nested / "ignored.txt").write_text('{"ignored": true}\n', encoding="utf-8")

    directory_records = list(common.iter_records(logs, suffixes=("__search.jsonl",)))
    assert directory_records == [("run/a__search.jsonl", {"a": 1}), ("run/a__search.jsonl", {"b": True})]

    archive = tmp_path / "logs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("z/run__search.jsonl", '{"z": 3}\n')
        zf.writestr("z/ignored.jsonl", '{"ignored": 1}\n')

    zip_records = list(common.iter_records(archive, suffixes=("__search.jsonl",)))
    assert zip_records == [("z/run__search.jsonl", {"z": 3})]


def test_common_conversion_helpers_and_output_writers(tmp_path: Path):
    assert common.to_float(True) == 1.0
    assert common.to_float("2.5") == 2.5
    assert common.to_float("bad", default=7.0) == 7.0
    assert common.to_int(False) == 0
    assert common.to_int("3.9") == 3
    assert common.to_int(None, default=4) == 4
    assert common.text_value(None, default="n/a") == "n/a"
    assert common.parse_allowed_modes("heuristic, hybrid", ("fallback",)) == {"heuristic", "hybrid"}
    assert common.parse_allowed_modes("", ("fallback",)) == {"fallback"}
    assert common.decision_key("r", "a", "5", 10, "search") == ("r", "a", 5, "10", "search")

    rows = [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]
    jsonl_path = tmp_path / "out" / "rows.jsonl"
    csv_path = tmp_path / "out" / "rows.csv"
    common.write_jsonl(jsonl_path, rows)
    common.write_csv(csv_path, rows)
    assert [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()] == rows
    assert "id,name" in csv_path.read_text(encoding="utf-8")

    empty_csv = tmp_path / "out" / "empty.csv"
    common.write_csv(empty_csv, [])
    assert empty_csv.exists()


def test_common_iter_records_rejects_unsupported_input(tmp_path: Path):
    with pytest.raises(ValueError):
        list(common.iter_records(tmp_path / "raw.txt"))
