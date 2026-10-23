import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json

import pandas as pd

from ticketclf.config import load_config, resolve
from ticketclf.evaluate import plot_history
from ticketclf.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="train again even if a saved model exists")
    args = parser.parse_args()

    cfg = load_config()
    model_dir = resolve(cfg["model"]["output_dir"])
    if args.retrain or not (model_dir / "config.json").exists():
        _, best_f1 = train(cfg)
        print("Best val macro-F1:", round(best_f1, 4))
    else:
        print("Saved model found, skipping training (use --retrain to train again).")

    reports = resolve(cfg["reports_dir"])
    hist = pd.DataFrame(json.loads((reports / "train_history.json").read_text(encoding="utf-8")))
    print(hist.round(4).to_string())
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    plot_history(hist, reports / "figures" / "training_curves.png")


if __name__ == "__main__":
    main()
