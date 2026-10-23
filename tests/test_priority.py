import pytest
import yaml

from ticketclf.config import resolve
from ticketclf.priority import PriorityMapper


def test_map_covers_77_unique_intents():
    data = yaml.safe_load(open(resolve("configs/priority_map.yaml"), encoding="utf-8"))
    intents = [i for group in data["priorities"].values() for i in group]
    assert len(intents) == 77
    assert len(set(intents)) == 77


def test_known_priorities():
    m = PriorityMapper.from_yaml()
    assert m.get("lost_or_stolen_card") == "High"
    assert m.get("exchange_rate") == "Low"
    assert m.get("pending_transfer") == "Medium"


def test_odd_banking77_labels_are_normalized():
    m = PriorityMapper.from_yaml()
    assert m.get("Refund_not_showing_up") == "Medium"
    assert m.get("reverted_card_payment?") == "Medium"


def test_unknown_intent_gets_default():
    assert PriorityMapper.from_yaml().get("something_new") == "Medium"


def test_invalid_level_and_duplicates_rejected():
    with pytest.raises(ValueError):
        PriorityMapper({"Urgent": ["a"]})
    with pytest.raises(ValueError):
        PriorityMapper({"High": ["a"], "Low": ["A"]})


def test_validate_reports_missing_and_unknown():
    m = PriorityMapper({"High": ["a"], "Low": ["b"]})
    res = m.validate(["a", "c"])
    assert res["missing"] == ["c"] and res["unknown"] == ["b"]