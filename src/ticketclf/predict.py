import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .config import load_config, resolve
from .priority import PriorityMapper


class TicketClassifier:
    
    def __init__(self, model_dir=None, cfg=None, device=None):
        self.cfg = cfg or load_config()
        model_dir = resolve(model_dir or self.cfg["model"]["output_dir"])
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found at {model_dir}. Train it first.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_length = self.cfg["model"]["max_length"]
        self.threshold = self.cfg["inference"]["confidence_threshold"]
        self.top_k = self.cfg["inference"]["top_k"]
        self.priority = PriorityMapper.from_yaml()

    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, texts, top_k=None) -> list:
        k = top_k or self.top_k
        results = []
        for p in self.predict_proba(texts):
            idx = np.argsort(p)[::-1][:k]
            top = [{"intent": self.id2label[int(i)], "confidence": round(float(p[i]), 4)} for i in idx]
            best = top[0]
            results.append({
                "intent": best["intent"],
                "confidence": best["confidence"],
                "priority": self.priority.get(best["intent"]),
                "needs_human_review": best["confidence"] < self.threshold,
                "top_k": top,
            })
        return results