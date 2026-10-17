"""Крок 2: протокол перенавчання (бюджети, seed, oracle, паритет OOF-епох)."""
import numpy as np
import pandas as pd
import pytest

from cleantext import data as data_mod
from cleantext import embeddings as emb_mod
from cleantext import pipeline as pipe_mod
from cleantext.config import BERT_PRESETS, PRESETS, ExperimentConfig
from cleantext.pipeline import run_experiment
from cleantext.runs import budget_view, load_run


# ---- конфігурація ---------------------------------------------------------------
def test_all_budgets_include_main_budget():
    cfg = ExperimentConfig(retrain_budgets=[0.10, 0.05], budget_frac=0.20)
    assert cfg.all_budgets() == [0.05, 0.10, 0.20]


def test_validate_rejects_big_budget():
    with pytest.raises(AssertionError):
        ExperimentConfig(retrain_budgets=[0.6]).validate()


def test_presets_give_cleanlab_same_epochs():
    for presets in (PRESETS, BERT_PRESETS):
        for p in presets.values():
            assert p["oof_epochs"] == p["epochs"]
    assert ExperimentConfig().oof_epochs == ExperimentConfig().epochs


def test_budget_view_old_format_unchanged():
    old = pd.DataFrame({"key": ["noisy", "proposed"], "macro_f1": [0.5, 0.6]})
    assert budget_view(old, 0.2).equals(old)


def test_budget_view_filters():
    cls = pd.DataFrame({"key": ["noisy", "proposed", "proposed"], "budget_frac": [np.nan, 0.1, 0.2],
                        "macro_f1": [0.5, 0.6, 0.7]})
    v = budget_view(cls, 0.2)
    assert v.key.tolist() == ["noisy", "proposed"] and v.macro_f1.tolist() == [0.5, 0.7]


# ---- наскрізна перевірка протоколу -------------------------------------------------
@pytest.mark.slow
def test_retrain_protocol(tmp_path, monkeypatch, corpus_df):
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path / "cache")
    calls = []
    real = pipe_mod.fit_model

    def spy(cfg, texts, y, num_labels, seed, epochs, track=False, val=None, progress=None, sample_weight=None):
        calls.append({"seed": seed, "epochs": epochs, "track": track, "n": len(texts)})
        return real(cfg, texts, y, num_labels, seed, epochs, track=track, val=val, progress=progress,
                    sample_weight=sample_weight)

    monkeypatch.setattr(pipe_mod, "fit_model", spy)
    cfg = ExperimentConfig(dataset="csv", n_train=200, n_val=50, n_test=80, noise_type="symmetric", noise_rate=0.05,
                           seed=11, embedding="tfidf_svd", epochs=2, oof_folds=2, oof_epochs=2, bootstrap=10,
                           k_neighbors=5, retrain_budgets=[0.05, 0.10], budget_frac=0.20, calibrate_weights=False)
    out = run_experiment(cfg, log=lambda *_: None, csv_df=corpus_df, out_dir=tmp_path / "run")

    # seed і епохи
    base = calls[0]
    assert base["track"] and base["seed"] == cfg.seed
    oof = [c for c in calls if c["seed"] != cfg.seed]
    assert len(oof) == cfg.oof_folds and all(c["epochs"] == cfg.oof_epochs for c in oof)
    retrain = [c for c in calls[1:] if c["seed"] == cfg.seed]
    assert len(retrain) == 3 * 7                      # 3 бюджети × (random, loss, proposed, weighted, cleanlab, aum, oracle)
    assert all(c["epochs"] == cfg.epochs for c in retrain)

    # таблиця класифікації
    cls = pd.read_csv(out / "classification_metrics.csv")
    n_noisy = int(np.load(out / "arrays.npz")["z"].sum())
    assert sorted(cls.budget_frac.dropna().unique()) == [0.05, 0.10, 0.20]
    assert {"val_macro_f1", "budget_mismatch", "delta_vs_random", "K"} <= set(cls.columns)
    assert cls.val_macro_f1.notna().all()
    for b, g in cls[cls.key != "noisy"].groupby("budget_frac"):
        k_b = max(1, round(200 * b))
        same_k = g[~g.key.isin(["oracle", "weighted"])]      # weighted не вилучає, лише знижує вагу
        assert (same_k.removed == k_b).all(), (b, same_k.removed.tolist())   # однаковий K для всіх стратегій
        orc = g[g.key == "oracle"].iloc[0]
        assert orc.removed == min(k_b, n_noisy) == orc.removed_corrupted      # oracle ≤ K і лише змінені
        assert bool(orc.budget_mismatch) == (k_b > n_noisy)
    assert cls[cls.key == "oracle"].budget_mismatch.any()                      # при η = 5 % і K = 20 % K > |z|

    # склад за класами
    rbc = pd.read_csv(out / "removed_by_class.csv")
    totals = rbc.groupby(["budget_frac", "key"])["count"].sum()
    for (b, key), n in totals.items():
        assert n == cls[(cls.budget_frac == b) & (cls.key == key)].removed.iloc[0]

    # читання запуску
    run = load_run(out)
    assert run.budgets == [0.05, 0.10, 0.20]
    assert set(run.classification_main.key) == {"noisy", "random", "loss", "proposed", "weighted", "cleanlab", "aum", "oracle"}
    assert set(run.removed_ids(0.10)) == {"random", "loss", "proposed", "cleanlab", "aum", "oracle"}
    assert len(run.removed_ids(0.10)["random"]) == 20
