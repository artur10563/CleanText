"""Повний AUM (Area Under the Margin; Pleiss et al., NeurIPS 2020) із пороговими зразками.

Реалізація за описом статті. Відмінності від авторського коду: оптимізатор і розклад lr — ті самі, що в базовій
моделі проєкту (AdamW, без розкладу); кількість епох — cfg.epochs; два прогони замість одного прогону + перенавчання.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .models import fit_model

AUM_SEED_OFFSET = 2000       # seed прогону r: cfg.seed + 2000 + r
MIN_THRESHOLD_SAMPLES = 5


@dataclass
class AumResult:
    aum: np.ndarray               # (N,) AUM кожного зразка, отримане тоді, коли він НЕ був пороговим
    flags: np.ndarray             # (N,) True — AUM нижче порогу свого прогону («імовірно помилкова мітка»)
    score: np.ndarray             # (N,) ранжувальна оцінка ризику: −AUM
    run_of: np.ndarray            # (N,) номер прогону, що оцінив зразок
    thresholds: list[float]       # поріг кожного прогону
    threshold_ids: list[np.ndarray]   # індекси порогових зразків кожного прогону (неперетинні)
    percentile: float

    def summary(self) -> dict:
        return {"percentile": self.percentile, "thresholds": [float(t) for t in self.thresholds],
                "n_threshold": [int(len(t)) for t in self.threshold_ids], "n_flagged": int(self.flags.sum()),
                "aum_mean": float(self.aum.mean())}


def aum_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUM(i) = (1/T) Σₜ [ z_{ỹᵢ}(t) − max_{c≠ỹᵢ} z_c(t) ]; logits: (T, N, K′), labels: (N,)."""
    t, n, _ = logits.shape
    ix = np.arange(n)
    assigned = logits[:, ix, labels]
    other = logits.copy()
    other[:, ix, labels] = -np.inf
    return (assigned - other.max(axis=2)).mean(axis=0)


def pick_threshold_sets(n: int, num_labels: int, seed: int, runs: int = 2) -> list[np.ndarray]:
    """runs неперетинних множин по N/(K+1) індексів (стаття: частка 1/(K+1) для фіктивного класу)."""
    size = n // (num_labels + 1)
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[r * size:(r + 1) * size]) for r in range(runs)]


def compute_aum(cfg: ExperimentConfig, texts, noisy_y: np.ndarray, num_labels: int, percentile: float = 99.0,
                progress=None, log=print) -> AumResult | None:
    """Два прогони з неперетинними пороговими множинами; кожен зразок оцінюється в прогоні, де він не порогова."""
    n = len(noisy_y)
    sets = pick_threshold_sets(n, num_labels, cfg.seed + AUM_SEED_OFFSET - 1)
    if len(sets[0]) < MIN_THRESHOLD_SAMPLES:
        log(f"AUM пропущено: порогових зразків {len(sets[0])} < {MIN_THRESHOLD_SAMPLES}")
        return None
    fake = num_labels                                    # індекс фіктивного класу K+1
    aum = np.full(n, np.nan)
    run_of = np.full(n, -1, dtype=int)
    thresholds: list[float] = []
    for r, thr_ids in enumerate(sets):
        labels = noisy_y.copy()
        labels[thr_ids] = fake
        fit = fit_model(cfg, texts, labels, num_labels + 1, cfg.seed + AUM_SEED_OFFSET + r, cfg.epochs,
                        track_logits=True, progress=(lambda e, m, r=r: progress((r + e / m) / len(sets))) if progress else None)
        a = aum_from_logits(fit.logits, labels)
        fit.release()
        thresholds.append(float(np.percentile(a[thr_ids], percentile)))
        eligible = np.ones(n, dtype=bool)
        eligible[thr_ids] = False                        # у цьому прогоні порогові зразки не оцінюються
        take = eligible & (run_of < 0)                   # прогін 0 бере всіх, крім множини 0; прогін 1 — решту (множину 0)
        aum[take] = a[take]
        run_of[take] = r
    assert (run_of >= 0).all() and not np.isnan(aum).any()
    flags = aum < np.asarray(thresholds)[run_of]
    return AumResult(aum, flags, -aum, run_of, thresholds, sets, percentile)
