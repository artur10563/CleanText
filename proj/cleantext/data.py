"""Завантаження корпусів, нормалізація, контроль дублікатів і розбиттів."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CACHE_DIR, ExperimentConfig

UA_NEWS_REPO = "FIdo-AI/ua-news"
UAREVIEWS_REPO = "KSE-RESEARCH-Group/UAReviews"
MIN_CALIB = 60      # менша калібрувальна вибірка не дає стабільної оцінки AP
# Пул текстів UA-News, який завантажується один раз і далі вибирається локально за seed.
NEWS_POOL_TRAIN = 15000
NEWS_POOL_TEST = 4000
NEWS_BODY_CHARS = 1200

INTENT_UA = {
    "Gratitude / Positive Feedback": "Подяка / позитивний відгук",
    "Complaint / Dissatisfaction": "Скарга / незадоволення",
    "Question / Request for Help": "Питання / прохання про допомогу",
    "Neutral Comment": "Нейтральний коментар",
    "Suggestion / Idea": "Пропозиція / ідея",
}
EMOTION_UA = {
    "Happiness": "Радість", "Anger": "Гнів", "Neutral": "Нейтрально", "Sadness": "Смуток",
    "Disgust": "Відраза", "Surprise": "Здивування", "Fear": "Страх",
}


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def digest(s: str) -> str:
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()


@dataclass
class Corpus:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]
    stats: dict = field(default_factory=dict)
    calib: pd.DataFrame | None = None      # калібрувальна підвибірка: неперетинна з train/val/test

    @property
    def num_labels(self) -> int:
        return len(self.labels)


def _download_news(split: str, limit: int, log) -> pd.DataFrame:
    path = CACHE_DIR / f"ua_news_{split}_{limit}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from datasets import load_dataset

    log(f"Завантаження UA-News ({split}, {limit} текстів) з Hugging Face…")
    stream = load_dataset(UA_NEWS_REPO, split=split, streaming=True)
    stream = stream.shuffle(seed=0, buffer_size=10000)
    rows = []
    for r in stream.take(limit):
        title = str(r.get("title") or "").strip()
        body = str(r.get("text") or "").strip()
        rows.append({"text": (title + ". " + body[:NEWS_BODY_CHARS]).strip(" ."),
                     "label": str(r.get("target") or "").strip()})
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _download_reviews(log) -> pd.DataFrame:
    path = CACHE_DIR / "uareviews_full.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from datasets import load_dataset

    log("Завантаження UAReviews з Hugging Face…")
    ds = load_dataset(UAREVIEWS_REPO, split="train")
    df = ds.to_pandas()[["id", "content", "final_category", "final_emotion", "split"]]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _clean(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df.text.str.len() >= min_len) & (~df.label.isin(["", "None", "nan", "NaN"]))]
    return df


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.sample(frac=1, random_state=seed)
    strat = df.label if df.label.value_counts().min() >= 2 else None
    part, _ = train_test_split(df, train_size=n, random_state=seed, stratify=strat)
    return part


def _split_train_val(pool: pd.DataFrame, n_train: int, n_val: int, seed: int):
    need = n_train + n_val
    if len(pool) < need:
        raise ValueError(f"Недостатньо текстів для train/val: є {len(pool)}, потрібно {need}. "
                         "Зменшіть обсяги вибірок.")
    pool = _sample(pool, need, seed)
    strat = pool.label if pool.label.value_counts().min() >= 2 else None
    return train_test_split(pool, train_size=n_train, test_size=n_val, random_state=seed, stratify=strat)


def remove_cross_split_duplicates(train, val, test):
    """Пріоритет test → val → train: оцінювальні приклади не потрапляють у навчання."""
    seen: set[str] = set()
    out, removed = [], {}
    for name, frame in [("test", test), ("val", val), ("train", train)]:
        df = frame.copy()
        df["hash"] = df.text.map(digest)
        before = len(df)
        df = df.drop_duplicates("hash")
        df = df[~df.hash.isin(seen)].copy()
        seen.update(df.hash)
        removed[name] = before - len(df)
        out.append(df.drop(columns="hash").reset_index(drop=True))
    test, val, train = out
    return train, val, test, removed


def build_calib(pool: pd.DataFrame, used: list[pd.DataFrame], labels: list[str], n: int, seed: int,
                log=print) -> pd.DataFrame | None:
    """Калібрувальна підвибірка з решти train-пулу: без текстів, що вже є в train/val/test."""
    if n <= 0:
        return None
    seen = set()
    for df in used:
        seen.update(df.text.map(digest))
    rest = pool.copy()
    rest["hash"] = rest.text.map(digest)
    rest = rest.drop_duplicates("hash")
    rest = rest[~rest.hash.isin(seen) & rest.label.isin(labels)].drop(columns="hash")
    if len(rest) < MIN_CALIB:
        log(f"Калібрувальну вибірку не сформовано: у пулі залишилось {len(rest)} < {MIN_CALIB} текстів")
        return None
    if len(rest) < n:
        log(f"Калібрувальна вибірка зменшена до {len(rest)} (запитано {n})")
    calib = _sample(rest, n, seed).reset_index(drop=True)
    calib["y"] = calib.label.map({l: i for i, l in enumerate(labels)}).astype(int)
    calib["id"] = np.arange(len(calib))
    calib["length"] = calib.text.str.len()
    return calib


def load_corpus(cfg: ExperimentConfig, log=print, csv_df: pd.DataFrame | None = None) -> Corpus:
    seed = cfg.seed
    if cfg.dataset == "ua_news":
        tr_pool = _clean(_download_news("train", NEWS_POOL_TRAIN, log), 20)
        te_pool = _clean(_download_news("test", NEWS_POOL_TEST, log), 20)
        source_info = {"repo": UA_NEWS_REPO, "pool_train": len(tr_pool), "pool_test": len(te_pool)}
    elif cfg.dataset == "uareviews":
        raw = _download_reviews(log)
        field_name = "final_category" if cfg.review_task == "intent" else "final_emotion"
        mapping = INTENT_UA if cfg.review_task == "intent" else EMOTION_UA
        df = pd.DataFrame({"text": raw.content, "label": raw[field_name].map(lambda v: mapping.get(v, v)),
                           "split": raw.split.astype(str).str.lower()})
        df = _clean(df, 5)
        # Поважаємо внутрішнє розбиття датасету; challenge не використовується в основному тесті.
        tr_pool = df[df.split.eq("train")].drop(columns="split")
        te_pool = df[df.split.eq("test")].drop(columns="split")
        source_info = {"repo": UAREVIEWS_REPO, "task": cfg.review_task,
                       "pool_train": len(tr_pool), "pool_test": len(te_pool)}
    else:
        if csv_df is None:
            if not cfg.csv_path:
                raise ValueError("Не вказано CSV-файл")
            csv_df = pd.read_csv(cfg.csv_path)
        df = _clean(csv_df[["text", "label"] + (["split"] if "split" in csv_df else [])], 5)
        if "split" in df and df.split.astype(str).str.lower().eq("test").any():
            sp = df.split.astype(str).str.lower()
            tr_pool, te_pool = df[sp.ne("test")].drop(columns="split"), df[sp.eq("test")].drop(columns="split")
        else:
            df = df.drop(columns="split", errors="ignore")
            strat = df.label if df.label.value_counts().min() >= 2 else None
            tr_pool, te_pool = train_test_split(df, test_size=min(cfg.n_test, len(df) // 4),
                                                random_state=seed, stratify=strat)
        source_info = {"repo": cfg.csv_path or "uploaded.csv", "pool_train": len(tr_pool), "pool_test": len(te_pool)}

    tr_pool = tr_pool.drop_duplicates("text")
    te_pool = te_pool.drop_duplicates("text")
    train, val = _split_train_val(tr_pool, cfg.n_train, cfg.n_val, seed)
    if len(te_pool) < cfg.n_test:
        raise ValueError(f"Тестовий пул замалий: {len(te_pool)} < {cfg.n_test}. Зменшіть N_TEST.")
    test = te_pool.sample(n=cfg.n_test, random_state=seed)
    train, val, test, removed = remove_cross_split_duplicates(train, val, test)

    labels = sorted(train.label.unique().tolist())
    for name, df in [("val", val), ("test", test)]:
        unknown = set(df.label) - set(labels)
        if unknown:  # рідкісні класи, відсутні в train, вилучаємо з оцінювання
            log(f"У {name} вилучено приклади невідомих класів: {unknown}")
    val = val[val.label.isin(labels)].reset_index(drop=True)
    test = test[test.label.isin(labels)].reset_index(drop=True)
    if len(train) < 50 or len(labels) < 2:
        raise ValueError("Недостатньо зразків або класів")

    label2id = {l: i for i, l in enumerate(labels)}
    for df in (train, val, test):
        df["y"] = df.label.map(label2id).astype(int)
        df["id"] = np.arange(len(df))
        df["length"] = df.text.str.len()

    calib = build_calib(tr_pool, [train, val, test], labels, cfg.n_calib if cfg.calibrate_weights else 0,
                        seed, log)

    stats = {
        "source": source_info,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "removed_duplicates": removed,
        "class_counts": {name: df.label.value_counts().to_dict() for name, df in
                         [("train", train), ("val", val), ("test", test)]},
        "length": {name: {"mean": float(df.length.mean()), "median": float(df.length.median())}
                   for name, df in [("train", train), ("val", val), ("test", test)]},
        "split_hashes": {name: hashlib.sha256("".join(df.text.map(digest)).encode()).hexdigest()[:16]
                         for name, df in [("train", train), ("val", val), ("test", test)]},
    }
    if calib is not None:
        stats["sizes"]["calib"] = len(calib)
        stats["class_counts"]["calib"] = calib.label.value_counts().to_dict()
        stats["split_hashes"]["calib"] = hashlib.sha256("".join(calib.text.map(digest)).encode()).hexdigest()[:16]
    return Corpus(train, val, test, labels, stats, calib)
