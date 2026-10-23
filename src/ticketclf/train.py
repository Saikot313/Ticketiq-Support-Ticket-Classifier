import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import resolve
from .data import load_splits
from .evaluate import compute_metrics, save_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


@torch.no_grad()
def predict_proba_loader(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
    return np.concatenate(out)


def train(cfg: dict):
    """Fine-tune DistilBERT with a custom PyTorch loop. Saves the best model (val macro-F1)."""
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    splits, label2id = load_splits(cfg)
    id2label = {i: l for l, i in label2id.items()}
    mcfg, tcfg = cfg["model"], cfg["train"]

    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["name"], num_labels=len(label2id), id2label=id2label, label2id=label2id
    ).to(device)

    collate = make_collate(tokenizer, mcfg["max_length"])
    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])
    train_loader = DataLoader(TextDataset(splits["train"]), batch_size=tcfg["batch_size"],
                              shuffle=True, collate_fn=collate, generator=gen)
    val_loader = DataLoader(TextDataset(splits["val"]), batch_size=tcfg["batch_size"] * 2,
                            shuffle=False, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": tcfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"])
    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * tcfg["warmup_ratio"]), total_steps)

    out_dir = resolve(mcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    y_val = splits["val"]["label"].to_numpy()
    best_f1, history = -1.0, []

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running, start = 0.0, time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{tcfg['epochs']}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()

        probs = predict_proba_loader(model, val_loader, device)
        m = compute_metrics(y_val, probs.argmax(axis=1), probs)
        row = {"epoch": epoch, "train_loss": running / len(train_loader),
               "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
               "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(row)

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  -> saved best model (val macro-F1 {best_f1:.4f})")

    reports = resolve(cfg["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    save_json(history, reports / "train_history.json")
    return history, best_f1