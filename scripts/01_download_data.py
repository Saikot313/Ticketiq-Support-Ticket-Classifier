import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from collections import Counter

import matplotlib.pyplot as plt

from ticketclf.config import load_config, resolve
from ticketclf.data import download_raw, load_splits, prepare
from ticketclf.priority import PriorityMapper


def main():
    cfg = load_config()
    download_raw(cfg)
    prepare(cfg)
    splits, label2id = load_splits(cfg)
    print("Splits:", {k: v.shape for k, v in splits.items()}, "| intents:", len(label2id))

    train = splits["train"]
    counts = train["category"].value_counts()
    lens = train["text"].str.split().str.len()
    print("\nSamples per intent (train):\n" + counts.describe().round(1).to_string())
    print("\nWords per query (train):\n" + lens.describe().round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts.plot.hist(bins=20, ax=axes[0], title="Samples per intent (train)")
    lens.plot.hist(bins=30, ax=axes[1], title="Words per query (train)")
    fig.tight_layout()
    figs = resolve(cfg["reports_dir"]) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "data_overview.png", dpi=150)

    mapper = PriorityMapper.from_yaml()
    check = mapper.validate(label2id.keys())
    assert not check["missing"] and not check["unknown"], f"Priority map mismatch: {check}"
    print("\nPriority map OK. Intents per level:", dict(Counter(mapper.get(l) for l in label2id)))


if __name__ == "__main__":
    main()
