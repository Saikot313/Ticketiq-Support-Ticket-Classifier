import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd

from ticketclf import evaluate as ev
from ticketclf.config import load_config, resolve
from ticketclf.data import load_splits
from ticketclf.predict import TicketClassifier
from ticketclf.priority import PriorityMapper


def main():
    cfg = load_config()
    reports = resolve(cfg["reports_dir"])
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    splits, label2id = load_splits(cfg)
    id2label = {i: l for l, i in label2id.items()}
    test = splits["test"]
    y_true = test["label"].to_numpy()

    # fine-tuned model vs baseline
    clf = TicketClassifier(cfg=cfg)
    probs = clf.predict_proba(test["text"].tolist())
    y_pred = probs.argmax(axis=1)
    bert = ev.compute_metrics(y_true, y_pred, probs)

    base = joblib.load(resolve(cfg["baseline"]["model_path"]))
    base_probs = base.predict_proba(test["text"])
    base_m = ev.compute_metrics(y_true, base_probs.argmax(axis=1), base_probs)

    comparison = pd.DataFrame({"TF-IDF + LogReg": base_m, "DistilBERT (fine-tuned)": bert}).T.round(4)
    print("=== Test set comparison ===")
    print(comparison.to_string())
    ev.plot_model_comparison(comparison, figs / "model_comparison.png")

    # where does it fail?
    report = ev.per_class_report(y_true, y_pred, id2label)
    print("\n=== Weakest intents ===")
    print(report.head(10).round(3).to_string())
    confs = ev.top_confusions(y_true, y_pred, id2label, n=12)
    print("\n=== Top confusions ===")
    print(confs.to_string())
    ev.plot_top_confusions(confs, figs / "top_confusions.png")
    ev.plot_confusion_matrix(y_true, y_pred, id2label, figs / "confusion_matrix.png")

    # business metrics
    mapper = PriorityMapper.from_yaml()
    prio = ev.priority_report(y_true, y_pred, id2label, mapper)
    cov = ev.coverage_accuracy(y_true, probs)
    print("\n=== Priority layer ===")
    print(prio)
    print("\n=== Confidence threshold vs coverage/accuracy ===")
    print(cov.round(4).to_string())

    # error analysis
    errors = test.assign(predicted=[id2label[i] for i in y_pred], confidence=probs.max(axis=1).round(4))
    errors = errors[y_pred != y_true][["text", "category", "predicted", "confidence"]]
    errors = errors.sort_values("confidence", ascending=False)
    errors.to_csv(reports / "error_analysis.csv", index=False)
    print("\n=== Most confident mistakes ===")
    print(errors.head(10).to_string())

    ev.save_json({
        "test_metrics": {"distilbert": bert, "baseline": base_m},
        "priority": prio,
        "coverage_vs_accuracy": cov.to_dict("records"),
    }, reports / "metrics.json")

    print("\n=== Paste into README ===")
    print("| Model | Accuracy | Macro-F1 | Top-3 acc. |")
    print("|---|---|---|---|")
    for name, m in (("TF-IDF + Logistic Regression (baseline)", base_m), ("DistilBERT (fine-tuned)", bert)):
        print(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['top3_accuracy']:.4f} |")
    high_recall = prio["high_priority_recall"]
    print(f"\nPriority accuracy: {prio['priority_accuracy']:.4f} | High-priority recall: "
          + ("n/a" if high_recall is None else f"{high_recall:.4f}"))


if __name__ == "__main__":
    main()
