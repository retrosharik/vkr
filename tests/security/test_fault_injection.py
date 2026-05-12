from __future__ import annotations

import json

import pytest
from rcrscore.entities import EntityID

from BaseRescueAgent.module.util.decision_logger import entity_value
from BaseRescueAgent.module.util.rescue_support import area_distance, get_area_id, is_rescuable_civilian
from tests.conftest import FakeCivilian, FakeEntity, FakeRefuge, FakeWorldInfo


def eid(value: int) -> EntityID:
    return EntityID(value)


@pytest.mark.security
def test_entity_value_is_safe_for_broken_entity_id_objects():
    class BrokenId:
        def get_value(self):
            raise RuntimeError("boom")

        def __str__(self):
            return "broken-id"

    assert entity_value(BrokenId()) == "broken-id"
    assert entity_value(None) == "None"


@pytest.mark.security
def test_area_distance_returns_infinity_for_unknown_or_invalid_entities():
    world = FakeWorldInfo([FakeEntity(1)])

    assert area_distance(world, eid(1), eid(999)) == float("inf")
    assert area_distance(world, None, eid(1)) == float("inf")
    assert get_area_id(world, eid(999)) is None


@pytest.mark.security
def test_rescue_filter_rejects_dead_or_refuge_civilians():
    refuge = FakeRefuge(10)
    dead = FakeCivilian(100, position=10, hp=0, damage=10, buriedness=0)
    stable = FakeCivilian(101, position=10, hp=10000, damage=0, buriedness=0)
    world = FakeWorldInfo([refuge, dead, stable])

    assert not is_rescuable_civilian(world, dead)
    assert not is_rescuable_civilian(world, stable)


@pytest.mark.security
def test_jsonl_log_parser_rejects_malformed_lines_without_silent_success(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    records = []
    with pytest.raises(json.JSONDecodeError):
        for line in path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))

    assert records == [{"ok": True}]
