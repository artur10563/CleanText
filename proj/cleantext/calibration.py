"""Калібрування ваг α, β, γ на окремій підвибірці з власним шумом.

Функція не отримує ні train, ні його індикатор змін z, ні test: тільки `calib`, конфігурацію моделі та тип шуму.
Тому підбір ваг не може використати цільову інформацію основного сценарію (docx 4.3).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import NOISE_TYPES, ExperimentConfig
from .embeddings import compute_embeddings, nearest_neighbours
from .models import fit_model
from .noise import add_noise
from .scoring import combine, risk_components

TIE_EPS = 1e-9
CALIB_FIT_SEED_OFFSET = 999      # як у v2: модель калібрування навчається з seed = CALIB_SEED + 999


@dataclass
class CalibResult:
    weights: tuple[float, float, float]      # (α, β, γ)
    ap: float                                # AP на прапорцях calib для знайдених ваг
    fixed_ap: float                          # AP на calib для апріорних ваг
    table: pd.DataFrame                      # alpha, beta, gamma, AP — вся сітка
    info: dict

    def to_json(self) -> dict:
        a, b, g = self.weights
        return {"alpha": a, "beta": b, "gamma": g, "ap_calib": self.ap, "ap_calib_fixed": self.fixed_ap, **self.info}


def simplex_grid(step: float) -> list[tuple[float, float, float]]:
    """Усі (α, β, γ) з кроком step на симплексі α + β + γ = 1."""
    n = int(round(1 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append((round(i * step, 4), round(j * step, 4), round((n - i - j) * step, 4)))
    return pts


def calib_noise(clean_y: np.ndarray, mode: str, kind: str, rate: float, seed: int, num_labels: int,
                neighbours: np.ndarray):
    """Шум на calib. known_type — один тип; mixed — по третині зразків кожного типу (випадкове розбиття)."""
    if mode == "known_type":
        return add_noise(clean_y, kind, rate, seed, num_labels, neighbours)
    kinds = list(NOISE_TYPES)
    groups = np.array_split(np.random.default_rng(seed + 31).permutation(len(clean_y)), len(kinds))
    noisy = np.asarray(clean_y, dtype=int).copy()
    for g, k in zip(groups, kinds):
        y_k, _, _ = add_noise(clean_y, k, rate, seed + 1 + kinds.index(k), num_labels, neighbours)
        noisy[g] = y_k[g]
    z = (noisy != clean_y).astype(int)
    trans = np.zeros((num_labels, num_labels), dtype=int)
    np.add.at(trans, (clean_y, noisy), 1)
    return noisy, z, trans


def _cache_key(cfg: ExperimentConfig, calib: pd.DataFrame, num_labels: int) -> str:
    parts = [cfg.classifier, cfg.embedding, cfg.epochs, cfg.hidden, cfg.dropout, cfg.lr, cfg.batch_size, cfg.max_features,
             cfg.bert_lr, cfg.bert_batch_size, cfg.max_length, cfg.k_neighbors, cfg.calib_mode, cfg.calib_noise_rate,
             cfg.calib_seed, cfg.coeff_grid_step, cfg.alpha, cfg.beta, cfg.gamma, num_labels,
             cfg.noise_type if cfg.calib_mode == "known_type" else "mixed",
             hashlib.sha256("\n".join(calib.text).encode("utf-8")).hexdigest()[:16],
             hashlib.sha256(",".join(map(str, calib.y)).encode()).hexdigest()[:16]]
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]


def best_weights(table: pd.DataFrame) -> pd.Series:
    """Максимум AP; тай-брейк — менше γ (як у v2), далі для детермінізму менше β, менше α."""
    top = table[table.AP >= table.AP.max() - TIE_EPS]
    return top.sort_values(["gamma", "beta", "alpha"]).iloc[0]


def calibrate_weights(cfg: ExperimentConfig, calib: pd.DataFrame, num_labels: int, log=print,
                      cache_dir: Path | None = None, progress=None) -> CalibResult:
    """Підбирає (α, β, γ) за AP на прапорцях калібрувальної підвибірки (вузли сітки + апріорна точка).

    Залежить лише від calib, типу шуму (режим known_type), гіперпараметрів моделі та calib_*; не залежить від
    cfg.seed, cfg.noise_rate і всього, що стосується основного train.
    """
    key = _cache_key(cfg, calib, num_labels)
    path = cache_dir / f"calib_{key}.json" if cache_dir else None
    if path is not None and path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        table = pd.read_csv(path.with_suffix(".csv"))
        info = {k: v for k, v in d.items() if k not in ("alpha", "beta", "gamma", "ap_calib", "ap_calib_fixed")}
        info["from_cache"] = True
        return CalibResult((d["alpha"], d["beta"], d["gamma"]), d["ap_calib"], d["ap_calib_fixed"], table, info)

    texts = calib.text.tolist()
    clean = calib.y.to_numpy()
    emb = compute_embeddings(texts, cfg.embedding, log=log)
    nb = nearest_neighbours(emb, max(cfg.k_neighbors + 1, 35))
    noisy, z, _ = calib_noise(clean, cfg.calib_mode, cfg.noise_type, cfg.calib_noise_rate, cfg.calib_seed,
                              num_labels, nb)
    fit = fit_model(cfg, texts, noisy, num_labels, cfg.calib_seed + CALIB_FIT_SEED_OFFSET, cfg.epochs, track=True, progress=progress)
    history = fit.history.astype(np.float32)
    fit.release()
    c, d, s = risk_components(history, noisy, nb, cfg.k_neighbors)

    fixed = (round(cfg.alpha, 4), round(cfg.beta, 4), round(cfg.gamma, 4))
    # як у v2: кандидати — вузли сітки плюс апріорна точка; обирається максимум AP серед усіх
    rows = [{"alpha": a, "beta": b, "gamma": g, "AP": float(average_precision_score(z, combine(c, d, s, a, b, g)))}
            for a, b, g in simplex_grid(cfg.coeff_grid_step) + [fixed]]
    table = pd.DataFrame(rows).drop_duplicates(["alpha", "beta", "gamma"]).reset_index(drop=True)
    fixed_ap = float(table[(table.alpha == fixed[0]) & (table.beta == fixed[1]) & (table.gamma == fixed[2])].AP.iloc[0])
    best = best_weights(table)
    info = {"mode": cfg.calib_mode, "noise_type": cfg.noise_type if cfg.calib_mode == "known_type" else "mixed",
            "noise_rate": cfg.calib_noise_rate, "seed": cfg.calib_seed, "n_calib": int(len(calib)),
            "n_corrupted": int(z.sum()), "grid_step": cfg.coeff_grid_step, "epochs": cfg.epochs,
            "classifier": cfg.classifier, "from_cache": False}
    res = CalibResult((float(best.alpha), float(best.beta), float(best.gamma)), float(best.AP), fixed_ap, table, info)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path.with_suffix(".csv"), index=False)
        path.write_text(json.dumps(res.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Калібрування: α = {res.weights[0]:.2f}, β = {res.weights[1]:.2f}, γ = {res.weights[2]:.2f} "
        f"(AP на calib = {res.ap:.3f}; апріорні ваги — {fixed_ap:.3f})")
    return res
