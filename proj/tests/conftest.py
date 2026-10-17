"""Спільні фікстури: синтетичний україномовний корпус без мережі."""
import numpy as np
import pandas as pd
import pytest

THEMES = {
    "політика": ["уряд", "парламент", "вибори", "закон", "президент", "депутат", "коаліція", "реформа"],
    "спорт": ["матч", "команда", "тренер", "чемпіонат", "гол", "турнір", "стадіон", "олімпіада"],
    "економіка": ["інфляція", "бюджет", "податок", "банк", "ринок", "експорт", "курс", "інвестиції"],
    "наука": ["дослідження", "вчені", "відкриття", "лабораторія", "космос", "експеримент", "теорія", "фізика"],
}


def make_corpus_df(per_class: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, words in THEMES.items():
        for i in range(per_class):
            picked = rng.choice(words, size=5, replace=True).tolist()
            rows.append({"text": f"{' '.join(picked)} новина номер {label[:3]}{i}", "label": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="session")
def corpus_df():
    return make_corpus_df()


@pytest.fixture
def small_labels():
    """Чисті мітки: 4 класи по 25 зразків."""
    return np.repeat(np.arange(4), 25)


@pytest.fixture
def ring_neighbours(small_labels):
    """Детерміновані «сусіди»: спершу зразки чужого класу, щоб semantic-шум був перевіряємим."""
    n = len(small_labels)
    return np.array([[(i + 25 + j) % n for j in range(5)] for i in range(n)])


@pytest.fixture(scope="session")
def big_corpus_df():
    """1000 текстів: вистачає на train 200 + val 50 + test 80 + calib 200."""
    return make_corpus_df(per_class=250)


@pytest.fixture(scope="session")
def run_variants(tmp_path_factory, big_corpus_df):
    """Чотири реальні запуски конвеєра різної конфігурації (спільні для тестів артефактів і UI)."""
    from cleantext import data as data_mod
    from cleantext import embeddings as emb_mod
    from cleantext.config import ExperimentConfig
    from cleantext.pipeline import run_experiment

    root = tmp_path_factory.mktemp("variants")
    mp = pytest.MonkeyPatch()
    mp.setattr(data_mod, "CACHE_DIR", root / "cache")
    mp.setattr(emb_mod, "CACHE_DIR", root / "cache")
    base = dict(dataset="csv", n_train=200, n_val=50, n_test=80, n_calib=200, noise_type="semantic", noise_rate=0.2,
                seed=7, embedding="tfidf_svd", epochs=2, oof_folds=2, oof_epochs=2, bootstrap=10, k_neighbors=5,
                retrain_budgets=[0.10], budget_frac=0.10, coeff_grid_step=0.25)
    variants = {
        "full": {},
        "no_calibration": dict(calibrate_weights=False),
        "detection_only": dict(run_retrain=False, run_cleanlab=False),
        "zero_noise": dict(noise_rate=0.0, run_retrain=False, run_cleanlab=False),
    }
    out = {}
    try:
        for name, kw in variants.items():
            out[name] = run_experiment(ExperimentConfig(**{**base, **kw}), log=lambda *_: None,
                                       csv_df=big_corpus_df, out_dir=root / name)
    finally:
        mp.undo()
    return out
