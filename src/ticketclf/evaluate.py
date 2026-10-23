import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, proba=None, k: int = 3) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    if proba is not None:
        topk = np.argsort(proba, axis=1)[:, -k:]
        out[f"top{k}_accuracy"] = float((topk == y_true[:, None]).any(axis=1).mean())
    return out


def per_class_report(y_true, y_pred, id2label) -> pd.DataFrame:
    labels = sorted(id2label)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    df = pd.DataFrame({"intent": [id2label[i] for i in labels],
                       "precision": p, "recall": r, "f1": f, "support": s})
    return df.sort_values("f1").reset_index(drop=True)


def top_confusions(y_true, y_pred, id2label, n: int = 15) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=sorted(id2label))
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm.ravel())[::-1][:n]
    rows = []
    for flat in order:
        i, j = np.unravel_index(flat, cm.shape)
        if cm[i, j] > 0:
            rows.append({"true": id2label[i], "predicted": id2label[j], "count": int(cm[i, j])})
    return pd.DataFrame(rows)


def priority_report(y_true, y_pred, id2label, mapper) -> dict:
    t = np.array([mapper.get(id2label[i]) for i in y_true])
    p = np.array([mapper.get(id2label[i]) for i in y_pred])
    high_t, high_p = t == "High", p == "High"
    return {
        "priority_accuracy": float((t == p).mean()),
        "high_priority_recall": float((p[high_t] == "High").mean()) if high_t.any() else None,
        "high_priority_precision": float((t[high_p] == "High").mean()) if high_p.any() else None,
    }


def coverage_accuracy(y_true, proba, thresholds=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    rows = []
    for t in thresholds:
        mask = conf >= t
        acc = float((pred[mask] == y_true[mask]).mean()) if mask.any() else None
        rows.append({"threshold": t, "coverage": float(mask.mean()), "accuracy": acc})
    return pd.DataFrame(rows)


def save_json(obj, path) -> None:
    path = str(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_confusion_matrix(y_true, y_pred, id2label, path):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(id2label), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted intent id")
    ax.set_ylabel("True intent id")
    ax.set_title("Normalized confusion matrix (77 intents)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_top_confusions(df, path):
    labels = [f"{a} → {b}" for a, b in zip(df["true"], df["predicted"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels[::-1], df["count"].tolist()[::-1], color="#4C72B0")
    ax.set_xlabel("Misclassified test samples")
    ax.set_title("Most frequent confusions (true → predicted)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_model_comparison(df, path):
    ax = df[["accuracy", "macro_f1", "top3_accuracy"]].T.plot.bar(figsize=(7, 4), rot=0)
    ax.set_ylim(0, 1.08)
    ax.set_title("Test set: baseline vs fine-tuned DistilBERT")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_history(hist: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(hist["epoch"], hist["val_macro_f1"], marker="o", label="val macro-F1")
    axes[1].plot(hist["epoch"], hist["val_accuracy"], marker="o", label="val accuracy")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig