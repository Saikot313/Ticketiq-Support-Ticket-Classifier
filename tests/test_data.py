import pandas as pd
import pytest

from ticketclf.data import load_splits, prepare


def _cfg(tmp_path):
    return {"seed": 0, "data": {"raw_dir": str(tmp_path / "raw"),
                                "processed_dir": str(tmp_path / "proc"),
                                "val_size": 0.25, "base_url": ""}}


def _write_raw(tmp_path, test_cats=("a", "b", "c", "d")):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [{"text": f"query {c} {i}", "category": c} for c in "abcd" for i in range(20)]
    pd.DataFrame(rows).to_csv(raw / "train.csv", index=False)
    pd.DataFrame([{"text": f"test {c}", "category": c} for c in test_cats]).to_csv(raw / "test.csv", index=False)


def test_prepare_creates_disjoint_stratified_splits(tmp_path):
    cfg = _cfg(tmp_path)
    _write_raw(tmp_path)
    label2id = prepare(cfg)
    splits, loaded = load_splits(cfg)
    assert label2id == loaded and len(label2id) == 4
    assert len(splits["train"]) == 60 and len(splits["val"]) == 20 and len(splits["test"]) == 4
    assert set(splits["train"]["text"]).isdisjoint(set(splits["val"]["text"]))
    assert set(splits["val"]["label"]) == {0, 1, 2, 3}


def test_unseen_test_label_raises(tmp_path):
    _write_raw(tmp_path, test_cats=("a", "zzz"))
    with pytest.raises(ValueError):
        prepare(_cfg(tmp_path))