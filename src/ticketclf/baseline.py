import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import resolve
from .data import load_splits
from .evaluate import compute_metrics


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=10.0)),
    ])


def train_baseline(cfg: dict):
    splits, _ = load_splits(cfg)
    pipe = build_pipeline()
    pipe.fit(splits["train"]["text"], splits["train"]["label"])

    results = {}
    for name in ("val", "test"):
        df = splits[name]
        proba = pipe.predict_proba(df["text"])
        results[name] = compute_metrics(df["label"].to_numpy(), proba.argmax(axis=1), proba)

    path = resolve(cfg["baseline"]["model_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return pipe, results