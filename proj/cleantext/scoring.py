"""Ознаки ризику C, D, S, інтегральна оцінка R та базові методи ранжування."""
from __future__ import annotations

import numpy as np

from .metrics import top_k

METHOD_NAMES = {
    "random": "Випадкове ранжування",
    "loss": "Loss (остання епоха)",
    "margin": "AUM-подібний margin (без порогових зразків)",
    "aum": "AUM (Pleiss et al., 2020, з пороговими зразками)",
    "cleanlab": "Cleanlab OOF (1 − self-confidence)",
    "dynamics": "Абляція: лише динаміка R(C,D,0)",
    "semantic": "Абляція: лише семантика R(0,0,S)",
    "proposed": "R фіксований (апріорні ваги)",
    "adaptive": "R адаптивний (ваги з калібрування)",
    "weighted": "R-зважування (м'яке, без вилучення)",
}
SHORT_NAMES = {
    "random": "Random", "loss": "Loss", "margin": "AUM-подібний", "aum": "AUM", "cleanlab": "Cleanlab OOF",
    "dynamics": "C+D", "semantic": "S", "proposed": "R фікс.", "adaptive": "R адапт.", "weighted": "Зважування", "oracle": "Oracle",
    "noisy": "Без очищення",
}


def risk_components(history: np.ndarray, y_obs: np.ndarray, neighbours: np.ndarray, k: int):
    """history: (T, N, K). Використовуються лише спостережувані мітки ỹ."""
    t, n, _ = history.shape
    ix = np.arange(n)
    c = 1 - history[:, ix, y_obs].mean(axis=0)
    d = 1 - (history.argmax(axis=2) == y_obs[None, :]).mean(axis=0)
    nb = neighbours[:, :k]
    s = 1 - (y_obs[nb] == y_obs[:, None]).mean(axis=1)
    return c, d, s.astype(float)


def margin_score(history: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    """Середня перевага найсильнішого альтернативного класу над заданим (діагностика, не AUM)."""
    ix = np.arange(history.shape[1])
    py = history[:, ix, y_obs]
    alt = history.copy()
    alt[:, ix, y_obs] = -np.inf
    return (alt.max(axis=2) - py).mean(axis=0)


def cartography(history: np.ndarray, y_obs: np.ndarray):
    """Dataset Cartography (Swayamdipta et al., 2020): confidence, variability, correctness за епохами."""
    ix = np.arange(history.shape[1])
    p = history[:, ix, y_obs]
    return p.mean(axis=0), p.std(axis=0), (history.argmax(axis=2) == y_obs[None, :]).mean(axis=0)


def combine(c, d, s, alpha, beta, gamma):
    return alpha * c + beta * d + gamma * s


def soft_weights(risk, k, w_min=0.2, power=2.0, scope="top_k"):
    """wᵢ = max{w_min, (1 − Rᵢ)^p} для K найризикованіших (scope=top_k, v2) або для всіх; решта wᵢ = 1.

    Повертає (ваги, індекси зразків із зниженою вагою).
    """
    risk = np.asarray(risk, dtype=float)
    picks = top_k(risk, k) if scope == "top_k" else np.arange(len(risk))
    w = np.ones(len(risk), dtype=np.float32)
    w[picks] = np.maximum(w_min, (1 - np.clip(risk[picks], 0, 1)) ** power)
    return w, picks


def all_scores(history, y_obs, neighbours, k, alpha, beta, gamma, seed, oof=None) -> dict:
    c, d, s = risk_components(history, y_obs, neighbours, k)
    ix = np.arange(len(y_obs))
    ab = alpha + beta if alpha + beta > 0 else 1.0
    scores = {
        "random": np.random.default_rng(seed + 777).random(len(y_obs)),
        "loss": -np.log(np.clip(history[-1, ix, y_obs], 1e-8, 1)),
        "margin": margin_score(history, y_obs),
        "dynamics": (alpha * c + beta * d) / ab,
        "semantic": s,
        "proposed": combine(c, d, s, alpha, beta, gamma),
    }
    if oof is not None:
        scores["cleanlab"] = cleanlab_score(y_obs, oof)
    return scores, {"C": c, "D": d, "S": s}


def cleanlab_score(y_obs, oof) -> np.ndarray:
    try:
        from cleanlab.rank import get_label_quality_scores
        return 1 - get_label_quality_scores(y_obs, oof, method="self_confidence")
    except ImportError:  # self-confidence без бібліотеки — те саме визначення
        return 1 - oof[np.arange(len(y_obs)), y_obs]


def cleanlab_flags(y_obs, oof) -> np.ndarray | None:
    try:
        from cleanlab.filter import find_label_issues
        return np.asarray(find_label_issues(labels=y_obs, pred_probs=oof), dtype=bool)
    except Exception:
        return None


def per_epoch_scores(history, y_obs, neighbours, k, alpha, beta, gamma):
    """Ранжування, обчислене лише за першими t епохами (аналіз раннього запам'ятовування)."""
    out = []
    ix = np.arange(len(y_obs))
    for t in range(1, history.shape[0] + 1):
        h = history[:t]
        c, d, s = risk_components(h, y_obs, neighbours, k)
        out.append({
            "epoch": t,
            "C": c, "D": d,
            "loss": -np.log(np.clip(h[-1, ix, y_obs], 1e-8, 1)),
            "proposed": combine(c, d, s, alpha, beta, gamma),
        })
    return out
