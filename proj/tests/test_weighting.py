"""Крок 4: м'яке зважування (sample_weight у навчанні та стратегія weighted у конвеєрі)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from cleantext import pipeline as pipe_mod
from cleantext.config import ExperimentConfig
from cleantext.models import _weighted_ce, fit_model
from cleantext.pipeline import run_experiment
from cleantext.scoring import soft_weights


# ---- формула ваг (v2, розд. 7) ---------------------------------------------------------
def test_soft_weights_top_k_matches_v2_formula():
    r = np.array([0.9, 0.1, 0.5, 0.7, 0.2, 1.4])           # 1.4 — поза [0; 1], має бути обрізано
    w, picks = soft_weights(r, 3, w_min=0.2, power=2.0, scope="top_k")
    assert set(picks.tolist()) == {5, 0, 3}
    assert w[5] == pytest.approx(0.2)                         # (1 − 1)^2 = 0 → w_min
    assert w[0] == pytest.approx(0.2)                         # (1 − 0.9)^2 = 0.01 → w_min
    assert w[3] == pytest.approx(max(0.2, (1 - 0.7) ** 2))    # 0.09 → w_min
    assert w[[1, 2, 4]].tolist() == [1.0, 1.0, 1.0]           # решта — вага 1


def test_soft_weights_graded_above_floor():
    r = np.array([0.5, 0.3, 0.1, 0.05])
    w, picks = soft_weights(r, 2, w_min=0.2, power=1.0, scope="top_k")
    assert w[0] == pytest.approx(0.5) and w[1] == pytest.approx(0.7) and w[2] == 1 and w[3] == 1


def test_soft_weights_bounds_and_scope_all():
    r = np.random.default_rng(1).random(50)
    w, picks = soft_weights(r, 10, 0.3, 2.0, "top_k")
    assert len(picks) == 10 and w.min() >= 0.3 and w.max() <= 1.0 and (w < 1).sum() <= 10
    wa, pa = soft_weights(r, 10, 0.3, 2.0, "all")
    assert len(pa) == 50 and wa.min() >= 0.3
    assert np.allclose(wa, np.maximum(0.3, (1 - r) ** 2))
    assert (wa <= 1).all()


# ---- функція втрат ---------------------------------------------------------------------
def test_weighted_ce_is_normalised_weighted_mean():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    y = torch.tensor([0, 0, 1])
    per = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    w = torch.tensor([1.0, 0.2, 0.5])
    assert _weighted_ce(logits, y, w).item() == pytest.approx(((w * per).sum() / w.sum()).item())
    ones = torch.ones(3)
    assert _weighted_ce(logits, y, ones).item() == pytest.approx(per.mean().item())
    # ваги, помножені на константу, не змінюють втрату (нормування на Σw)
    assert _weighted_ce(logits, y, w * 7).item() == pytest.approx(_weighted_ce(logits, y, w).item())


# ---- fit_model --------------------------------------------------------------------------
@pytest.fixture
def toy_fit(corpus_df):
    cfg = ExperimentConfig(epochs=3, hidden=32, batch_size=16, max_features=2000)
    df = corpus_df.iloc[:120]
    labels = sorted(df.label.unique())
    y = df.label.map({l: i for i, l in enumerate(labels)}).to_numpy()
    return cfg, df.text.tolist(), y, len(labels)


def test_unit_weights_equal_no_weights(toy_fit):
    cfg, texts, y, k = toy_fit
    a = fit_model(cfg, texts, y, k, 5, 3)
    b = fit_model(cfg, texts, y, k, 5, 3, sample_weight=np.ones(len(y)))
    pa, pb = a.predict_proba(texts), b.predict_proba(texts)
    assert np.allclose(pa, pb, atol=1e-4)


def test_nonuniform_weights_change_the_model(toy_fit):
    cfg, texts, y, k = toy_fit
    w = np.where(np.arange(len(y)) % 3 == 0, 0.05, 1.0)
    a = fit_model(cfg, texts, y, k, 5, 3)
    b = fit_model(cfg, texts, y, k, 5, 3, sample_weight=w)
    assert not np.allclose(a.predict_proba(texts), b.predict_proba(texts), atol=1e-4)


def test_invalid_weights_rejected(toy_fit):
    cfg, texts, y, k = toy_fit
    with pytest.raises(AssertionError):
        fit_model(cfg, texts, y, k, 5, 1, sample_weight=np.ones(len(y) - 1))
    with pytest.raises(AssertionError):
        fit_model(cfg, texts, y, k, 5, 1, sample_weight=-np.ones(len(y)))


def test_zero_weight_samples_do_not_drive_the_loss(toy_fit):
    """Ваги 0 для половини вибірки з навмисно хибними мітками: модель не повторює ці мітки."""
    cfg, texts, y, k = toy_fit
    bad = y.copy()
    bad[::2] = (bad[::2] + 1) % k
    w = np.where(np.arange(len(y)) % 2 == 0, 0.0, 1.0)
    cfg = ExperimentConfig(epochs=8, hidden=64, batch_size=16, max_features=2000, lr=3e-3)
    m = fit_model(cfg, texts, bad, k, 1, 8, sample_weight=w)
    acc_true = (m.predict_proba(texts).argmax(1) == y)[::2].mean()          # на «зіпсованих» позиціях
    m2 = fit_model(cfg, texts, bad, k, 1, 8)
    acc_unweighted = (m2.predict_proba(texts).argmax(1) == y)[::2].mean()
    assert acc_true > acc_unweighted


def test_config_validation_for_weighting():
    ExperimentConfig(weight_min=0.2, weight_power=2.0, weight_scope="all").validate()
    for bad in (dict(weight_min=0.0), dict(weight_min=1.5), dict(weight_power=0), dict(weight_scope="x")):
        with pytest.raises(AssertionError):
            ExperimentConfig(**bad).validate()


# ---- конвеєр ----------------------------------------------------------------------------
@pytest.mark.slow
def test_pipeline_weighted_rows(run_variants):
    out = run_variants["full"]
    cls = pd.read_csv(out / "classification_metrics.csv")
    w = cls[cls.key == "weighted"]
    assert len(w) == 1 and w.budget_frac.iloc[0] == 0.10                # по одному рядку на бюджет
    row = w.iloc[0]
    assert row.removed == 0 and row.removed_corrupted == 0
    assert row.n_train == cls[cls.key == "noisy"].n_train.iloc[0]       # нічого не вилучено
    assert row.downweighted == row.K
    assert 0.2 <= row.weight_mean < 1.0
    others = cls[~cls.key.isin(["weighted"])]
    assert (others.weight_mean == 1.0).all() and (others.downweighted == 0).all()

    # середня вага збігається з перерахунком за збереженим R (адаптивним, бо калібрування увімкнено)
    a = np.load(out / "arrays.npz")
    expected, picks = soft_weights(a["score_adaptive"], int(row.K), 0.2, 2.0, "top_k")
    assert row.weight_mean == pytest.approx(float(expected.mean()), abs=1e-6)
    m = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(m["downweighted_ids"]["0.10"]["weighted"]) == row.K
    assert "weighted" not in m["removed_ids"]["0.10"]                    # вилучених немає
    assert "pred_weighted" in a.files
    assert cls[cls.key == "weighted"].delta_vs_random.notna().all()


@pytest.mark.slow
def test_pipeline_weighted_uses_fixed_risk_without_calibration(run_variants):
    out = run_variants["no_calibration"]
    cls = pd.read_csv(out / "classification_metrics.csv")
    row = cls[cls.key == "weighted"].iloc[0]
    a = np.load(out / "arrays.npz")
    expected, _ = soft_weights(a["score_proposed"], int(row.K), 0.2, 2.0, "top_k")
    assert row.weight_mean == pytest.approx(float(expected.mean()), abs=1e-6)


@pytest.mark.slow
def test_weighted_fit_gets_full_train_set_and_same_hyperparameters(tmp_path, monkeypatch, big_corpus_df):
    from cleantext import data as data_mod
    from cleantext import embeddings as emb_mod
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path / "cache")
    calls = []
    real = pipe_mod.fit_model

    def spy(cfg, texts, y, num_labels, seed, epochs, track=False, val=None, progress=None, sample_weight=None):
        calls.append({"seed": seed, "epochs": epochs, "n": len(texts), "weighted": sample_weight is not None,
                      "track": track})
        return real(cfg, texts, y, num_labels, seed, epochs, track=track, val=val, progress=progress,
                    sample_weight=sample_weight)

    monkeypatch.setattr(pipe_mod, "fit_model", spy)
    cfg = ExperimentConfig(dataset="csv", n_train=200, n_val=50, n_test=80, noise_rate=0.2, seed=3, embedding="tfidf_svd",
                           epochs=2, oof_folds=2, oof_epochs=2, bootstrap=5, k_neighbors=5, retrain_budgets=[0.10],
                           budget_frac=0.10, calibrate_weights=False, run_cleanlab=False, run_oracle=False)
    run_experiment(cfg, log=lambda *_: None, csv_df=big_corpus_df, out_dir=tmp_path / "r")
    weighted = [c for c in calls if c["weighted"]]
    assert len(weighted) == 1 and weighted[0]["n"] == 200 and weighted[0]["seed"] == cfg.seed
    assert weighted[0]["epochs"] == cfg.epochs
    unweighted_retrain = [c for c in calls[1:] if not c["weighted"]]
    assert all(c["seed"] == cfg.seed for c in unweighted_retrain)


@pytest.mark.slow
def test_weighting_can_be_switched_off(tmp_path, monkeypatch, big_corpus_df):
    from cleantext import data as data_mod
    from cleantext import embeddings as emb_mod
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path / "cache")
    cfg = ExperimentConfig(dataset="csv", n_train=200, n_val=50, n_test=80, noise_rate=0.2, seed=3, embedding="tfidf_svd",
                           epochs=2, oof_folds=2, oof_epochs=2, bootstrap=5, k_neighbors=5, retrain_budgets=[0.10],
                           budget_frac=0.10, calibrate_weights=False, run_cleanlab=False, run_oracle=False,
                           run_weighted=False)
    out = run_experiment(cfg, log=lambda *_: None, csv_df=big_corpus_df, out_dir=tmp_path / "r")
    cls = pd.read_csv(out / "classification_metrics.csv")
    assert "weighted" not in set(cls.key)
