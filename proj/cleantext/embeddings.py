"""Заморожені семантичні подання текстів та індекс найближчих сусідів."""
from __future__ import annotations

import hashlib

import numpy as np

from .config import CACHE_DIR, MINILM_ID


def compute_embeddings(texts: list[str], method: str, log=print) -> np.ndarray:
    key = hashlib.sha256(("\n".join(texts) + method).encode("utf-8")).hexdigest()[:20]
    path = CACHE_DIR / f"emb_{method}_{key}.npy"
    if path.exists():
        return np.load(path)
    if method == "minilm":
        import torch
        from sentence_transformers import SentenceTransformer

        log("Обчислення семантичних подань MiniLM…")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enc = SentenceTransformer(MINILM_ID, device=device)
        emb = enc.encode(texts, batch_size=32, show_progress_bar=False,
                         normalize_embeddings=True).astype("float32")
        del enc
    else:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        log("Обчислення LSA-подань (TF-IDF + SVD)…")
        x = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), sublinear_tf=True,
                            min_df=1).fit_transform(texts)
        dim = max(2, min(256, x.shape[1] - 1, x.shape[0] - 1))
        emb = normalize(TruncatedSVD(dim, random_state=0).fit_transform(x)).astype("float32")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def nearest_neighbours(emb: np.ndarray, m: int, chunk: int = 2048) -> np.ndarray:
    """Індекси m найближчих (косинус) сусідів для кожного зразка; сам зразок виключено."""
    n = len(emb)
    m = min(m, n - 1)
    out = np.empty((n, m), dtype=np.int64)
    for s in range(0, n, chunk):
        sims = emb[s:s + chunk] @ emb.T
        rows = np.arange(sims.shape[0])
        sims[rows, rows + s] = -np.inf
        part = np.argpartition(-sims, m - 1, axis=1)[:, :m]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1, kind="stable")
        out[s:s + chunk] = np.take_along_axis(part, order, axis=1)
    return out
