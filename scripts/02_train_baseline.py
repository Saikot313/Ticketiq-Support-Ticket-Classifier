import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from ticketclf.baseline import train_baseline
from ticketclf.config import load_config, resolve
from ticketclf.evaluate import save_json


def main():
    cfg = load_config()
    _, results = train_baseline(cfg)
    reports = resolve(cfg["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    save_json(results, reports / "baseline_metrics.json")
    print("Baseline (TF-IDF + Logistic Regression):")
    print(pd.DataFrame(results).T.round(4).to_string())


if __name__ == "__main__":
    main()
