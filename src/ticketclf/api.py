from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import TicketClassifier

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["clf"] = TicketClassifier()
    except FileNotFoundError:
        state["clf"] = None
    yield
    state.clear()


app = FastAPI(
    title="TicketIQ",
    description="Support ticket intent classification (DistilBERT, Banking77) with priority routing.",
    version="1.0.0",
    lifespan=lifespan,
)


class TicketIn(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class IntentScore(BaseModel):
    intent: str
    confidence: float


class PredictionOut(BaseModel):
    intent: str
    confidence: float
    priority: str
    needs_human_review: bool
    top_k: list[IntentScore]


class BatchOut(BaseModel):
    count: int
    predictions: list[PredictionOut]


def _get_clf():
    clf = state.get("clf")
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    return clf


def _clean(texts):
    cleaned = [t.strip() for t in texts]
    if any(len(t) < 3 for t in cleaned):
        raise HTTPException(status_code=422, detail="Every text must have at least 3 characters.")
    return cleaned


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.get("clf") is not None}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TicketIn):
    clf = _get_clf()
    return clf.predict(_clean([payload.text]))[0]


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(payload: BatchIn):
    clf = _get_clf()
    preds = clf.predict(_clean(payload.texts))
    return {"count": len(preds), "predictions": preds}