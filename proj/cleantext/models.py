"""Нейромережеві класифікатори з реєстрацією ймовірностей після кожної епохи."""
from __future__ import annotations

import gc
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score

from .config import DISTILBERT_ID, ExperimentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class FitResult:
    predict_proba: callable
    history: np.ndarray | None      # (T, N, K): pₜ(c | xᵢ) для навчальних прикладів
    log: pd.DataFrame
    release: callable
    logits: np.ndarray | None = None  # (T, N, K): логіти після кожної епохи (лише при track_logits; для AUM)


def _softmax_eval(model, batches) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for xb in batches():
            out.append(torch.softmax(model(xb).float(), dim=-1).cpu().numpy())
    return np.vstack(out)


def _logits_eval(model, batches) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for xb in batches():
            out.append(model(xb).float().cpu().numpy())
    return np.vstack(out)


# ---------------------------------------------------------------- MLP на TF-IDF
class TfidfMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def _weighted_ce(logits, target, w):
    """Нормована зважена крос-ентропія L_w = Σ wᵢℓᵢ / Σ wᵢ (v2, розд. 4)."""
    return (w * F.cross_entropy(logits, target, reduction="none")).sum() / w.sum().clamp_min(1e-8)


def _fit_mlp(cfg, texts, y, num_labels, seed, epochs, track, val, progress, sample_weight=None, track_logits=False):
    from sklearn.feature_extraction.text import TfidfVectorizer

    set_seed(seed)
    vec = TfidfVectorizer(max_features=cfg.max_features, ngram_range=(1, 2), sublinear_tf=True,
                          min_df=1, dtype=np.float32)
    x = vec.fit_transform(texts).tocsr()
    model = TfidfMLP(x.shape[1], cfg.hidden, num_labels, cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    yt = torch.as_tensor(y, dtype=torch.long)
    wt = torch.as_tensor(sample_weight, dtype=torch.float32) if sample_weight is not None else None
    gen = np.random.default_rng(seed)

    def batches_of(mat):
        def it():
            for s in range(0, mat.shape[0], 256):
                yield torch.from_numpy(mat[s:s + 256].toarray()).to(DEVICE)
        return it

    def predict_proba(new_texts):
        return _softmax_eval(model, batches_of(vec.transform(new_texts).tocsr()))

    xv = vec.transform(val[0]).tocsr() if val is not None else None
    trace, log, ltrace = [], [], []
    for ep in range(epochs):
        model.train()
        perm = gen.permutation(x.shape[0])
        total, nb = 0.0, 0
        for s in range(0, len(perm), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            xb = torch.from_numpy(x[idx].toarray()).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = (loss_fn(logits, yt[idx].to(DEVICE)) if wt is None
                    else _weighted_ce(logits, yt[idx].to(DEVICE), wt[idx].to(DEVICE)))
            loss.backward()
            opt.step()
            total += float(loss.detach()); nb += 1
        rec = {"epoch": ep + 1, "train_loss": total / max(nb, 1)}
        if track:
            p = _softmax_eval(model, batches_of(x))
            trace.append(p)
            rec["train_acc_observed"] = float((p.argmax(1) == y).mean())
        if track_logits:
            ltrace.append(_logits_eval(model, batches_of(x)))
        if xv is not None:
            pv = _softmax_eval(model, batches_of(xv))
            rec["val_loss"] = float(-np.log(np.clip(pv[np.arange(len(val[1])), val[1]], 1e-8, 1)).mean())
            rec["val_macro_f1"] = f1_score(val[1], pv.argmax(1), average="macro", zero_division=0)
        log.append(rec)
        if progress:
            progress(ep + 1, epochs)

    def release():
        nonlocal model
        del model
        gc.collect()

    return FitResult(predict_proba, np.stack(trace) if track else None, pd.DataFrame(log), release,
                     np.stack(ltrace) if track_logits else None)


# ---------------------------------------------------------------- DistilBERT
def _fit_bert(cfg, texts, y, num_labels, seed, epochs, track, val, progress, sample_weight=None, track_logits=False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(DISTILBERT_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_ID, num_labels=num_labels, ignore_mismatched_sizes=True).to(DEVICE)

    def encode(tx):
        return tok(list(tx), truncation=True, max_length=cfg.max_length, padding="max_length",
                   return_tensors="pt")

    def batches_of(enc):
        def it():
            n = enc["input_ids"].shape[0]
            for s in range(0, n, cfg.bert_batch_size * 2):
                yield {k: v[s:s + cfg.bert_batch_size * 2].to(DEVICE) for k, v in enc.items()}
        return it

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, b):
            return self.m(**b).logits

    wrapped = Wrap(model)
    enc = encode(texts)
    yt = torch.as_tensor(y, dtype=torch.long)
    wt = torch.as_tensor(sample_weight, dtype=torch.float32) if sample_weight is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.bert_lr, weight_decay=0.01)
    gen = np.random.default_rng(seed)
    encv = encode(val[0]) if val is not None else None

    def predict_proba(new_texts):
        return _softmax_eval(wrapped, batches_of(encode(new_texts)))

    trace, log, ltrace = [], [], []
    for ep in range(epochs):
        model.train()
        perm = gen.permutation(len(y))
        total, nb = 0.0, 0
        for s in range(0, len(perm), cfg.bert_batch_size):
            idx = torch.as_tensor(perm[s:s + cfg.bert_batch_size])
            batch = {k: v[idx].to(DEVICE) for k, v in enc.items()}
            opt.zero_grad(set_to_none=True)
            if wt is None:
                loss = model(**batch, labels=yt[idx].to(DEVICE)).loss
            else:
                loss = _weighted_ce(model(**batch).logits, yt[idx].to(DEVICE), wt[idx].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()); nb += 1
            if progress:
                progress(ep + (s + len(idx)) / len(perm), epochs)
        rec = {"epoch": ep + 1, "train_loss": total / max(nb, 1)}
        if track:
            p = _softmax_eval(wrapped, batches_of(enc))
            trace.append(p)
            rec["train_acc_observed"] = float((p.argmax(1) == y).mean())
        if track_logits:
            ltrace.append(_logits_eval(wrapped, batches_of(enc)))
        if encv is not None:
            pv = _softmax_eval(wrapped, batches_of(encv))
            rec["val_loss"] = float(-np.log(np.clip(pv[np.arange(len(val[1])), val[1]], 1e-8, 1)).mean())
            rec["val_macro_f1"] = f1_score(val[1], pv.argmax(1), average="macro", zero_division=0)
        log.append(rec)

    def release():
        nonlocal model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return FitResult(predict_proba, np.stack(trace) if track else None, pd.DataFrame(log), release,
                     np.stack(ltrace) if track_logits else None)


def fit_model(cfg: ExperimentConfig, texts, y, num_labels: int, seed: int, epochs: int,
              track: bool = False, val=None, progress=None, sample_weight=None, track_logits: bool = False) -> FitResult:
    """sample_weight=None — звичайна крос-ентропія; інакше нормована зважена: Σ wᵢℓᵢ / Σ wᵢ за батчем."""
    y = np.asarray(y, dtype=int)
    texts = list(texts)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        assert sample_weight.shape == y.shape and (sample_weight >= 0).all(), "некоректні ваги зразків"
    fn = _fit_bert if cfg.classifier == "distilbert" else _fit_mlp
    return fn(cfg, texts, y, num_labels, seed, epochs, track, val, progress, sample_weight, track_logits)
