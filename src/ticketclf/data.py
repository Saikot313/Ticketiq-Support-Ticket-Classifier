import json
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve

RAW_FILES = ("train.csv", "test.csv")


def download_raw(cfg: dict, force: bool = False):
    raw_dir = resolve(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        dest = raw_dir / name
        if dest.exists() and not force:
            continue
        urllib.request.urlretrieve(cfg["data"]["base_url"] + name, dest)
    return raw_dir


def prepare(cfg: dict) -> dict:
    
    raw_dir = resolve(cfg["data"]["raw_dir"])
    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    full_train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")

    labels = sorted(full_train["category"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    unseen = set(test["category"]) - set(label2id)
    if unseen:
        raise ValueError(f"Test set has labels not in train: {sorted(unseen)}")

    train, val = train_test_split(
        full_train,
        test_size=cfg["data"]["val_size"],
        stratify=full_train["category"],
        random_state=cfg["seed"],
    )
    for name, df in (("train", train), ("val", val), ("test", test)):
        df = df[["text", "category"]].copy()
        df["label"] = df["category"].map(label2id)
        df.reset_index(drop=True).to_csv(out_dir / f"{name}.csv", index=False)

    (out_dir / "label_map.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
    return label2id


def load_splits(cfg: dict):
    out_dir = resolve(cfg["data"]["processed_dir"])
    splits = {n: pd.read_csv(out_dir / f"{n}.csv") for n in ("train", "val", "test")}
    label2id = json.loads((out_dir / "label_map.json").read_text(encoding="utf-8"))
    return splits, label2id