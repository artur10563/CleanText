"""Метрики виявлення некоректних міток і класифікації, бутстреп-інтервали."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, auc, average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)


def top_k(score, k):
    return np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]


def detection_metrics(z, score, k) -> dict:
    z = np.asarray(z, dtype=int)
    score = np.asarray(score, dtype=float)
    tp = int(z[top_k(score, k)].sum())
    both = len(np.unique(z)) == 2
    if both:
        p, r, _ = precision_recall_curve(z, score)
        pr_auc = auc(r, p)
    fp = k - tp
    fn = int(z.sum()) - tp
    prec = tp / max(k, 1)
    rec = tp / z.sum() if z.sum() else np.nan
    return {
        "AP": average_precision_score(z, score) if both else np.nan,
        "PR-AUC": pr_auc if both else np.nan,
        "ROC-AUC": roc_auc_score(z, score) if both else np.nan,
        "Precision@K": prec,
        "Recall@K": rec,
        "F1@K": 2 * prec * rec / (prec + rec) if both and prec + rec > 0 else np.nan,
        "TP": tp, "FP": fp, "FN": fn, "K": k,
    }


def pr_curve(z, score, max_points: int = 300):
    p, r, _ = precision_recall_curve(z, score)
    if len(p) > max_points:
        idx = np.linspace(0, len(p) - 1, max_points).astype(int)
        p, r = p[idx], r[idx]
    return r, p


def precision_at_curve(z, score, fracs):
    z = np.asarray(z)
    order = top_k(score, len(z))
    out = []
    for f in fracs:
        k = max(1, int(round(len(z) * f)))
        tp = z[order[:k]].sum()
        out.append({"frac": f, "K": k, "Precision@K": tp / k,
                    "Recall@K": tp / z.sum() if z.sum() else np.nan})
    return out


def bootstrap_ap_diff(z, s1, s2, n_boot=500, seed=0):
    """Стратифікований бутстреп за зразками: різниця AP(s1) − AP(s2) і 95 % CI."""
    z = np.asarray(z)
    if len(np.unique(z)) < 2 or n_boot <= 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    pos, neg = np.where(z == 1)[0], np.where(z == 0)[0]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(average_precision_score(z[idx], s1[idx]) - average_precision_score(z[idx], s2[idx]))
    diffs = np.asarray(diffs)
    return float(average_precision_score(z, s1) - average_precision_score(z, s2)), \
        float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def clf_metrics(y_true, probs) -> dict:
    pred = probs.argmax(axis=1)
    return {"accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0)}


def bootstrap_f1_diff(y_true, pred_a, pred_b, n_boot=500, seed=0):
    """Парний бутстреп на однакових індексах тестових текстів: ΔMacro-F1 (a − b)."""
    y_true = np.asarray(y_true)
    if n_boot <= 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
                     - f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0))
    point = f1_score(y_true, pred_a, average="macro", zero_division=0) - \
        f1_score(y_true, pred_b, average="macro", zero_division=0)
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
