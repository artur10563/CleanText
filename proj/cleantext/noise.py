"""Контрольоване внесення шуму в мітки навчальної вибірки (лише train)."""
from __future__ import annotations

import numpy as np


def add_noise(clean: np.ndarray, kind: str, rate: float, seed: int, num_labels: int,
              neighbours: np.ndarray | None = None):
    """Повертає (зашумлені мітки ỹ, індикатор z = 1[ỹ ≠ y⁰], матрицю переходів)."""
    rng = np.random.default_rng(seed)
    clean = np.asarray(clean, dtype=int)
    y = clean.copy()
    n = len(y)
    m = int(round(n * rate))
    for i in rng.choice(n, size=m, replace=False):
        prev = int(clean[i])
        if kind == "symmetric":
            y[i] = int(rng.choice([v for v in range(num_labels) if v != prev]))
        elif kind == "asymmetric":
            y[i] = (prev + 1) % num_labels  # фіксована відтворювана матриця переходів
        elif kind == "semantic":
            # мітка найближчого семантично тексту ІНШОГО класу (лише симуляція)
            opts = [j for j in neighbours[i] if clean[j] != prev] if neighbours is not None else []
            y[i] = int(clean[opts[0]]) if opts else int(rng.choice([v for v in range(num_labels) if v != prev]))
        else:
            raise ValueError(kind)
    z = (y != clean).astype(int)
    assert z.sum() == m
    transition = np.zeros((num_labels, num_labels), dtype=int)
    np.add.at(transition, (clean, y), 1)
    return y, z, transition
