"""Крок 5: повний AUM (Pleiss et al., 2020) і Dataset Cartography."""
import json

import numpy as np
import pandas as pd
import pytest

from cleantext import aum as aum_mod
from cleantext.aum import aum_from_logits, compute_aum, pick_threshold_sets
from cleantext.config import ExperimentConfig
from cleantext.models import fit_model
from cleantext.scoring import cartography


# ---- формула AUM на штучних логітах -----------------------------------------------------
def test_aum_from_logits_known_values():
    # T = 2 епохи, N = 2 зразки, 3 класи
    logits = np.array([[[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]],
                       [[4.0, 1.0, 0.0], [1.0, 1.0, 2.0]]])
    labels = np.array([0, 1])
    # зразок 0: (2 − 1) і (4 − 1) → 2; зразок 1: (3 − 1) і (1 − 2) → 0.5
    assert aum_from_logits(logits, labels).tolist() == pytest.approx([2.0, 0.5])


def test_aum_naive_loop_agrees_on_random_logits():
    rng = np.random.default_rng(0)
    t, n, k = 5, 20, 4
    logits = rng.normal(size=(t, n, k))
    y = rng.integers(0, k, n)
    expect = [np.mean([logits[e, i, y[i]] - max(logits[e, i, c] for c in range(k) if c != y[i]) for e in range(t)])
              for i in range(n)]
    assert aum_from_logits(logits, y) == pytest.approx(expect)


def test_mislabeled_sample_has_lower_aum():
    logits = np.zeros((3, 2, 3))
    logits[:, :, 0] = 5.0                       # модель впевнена в класі 0
    assert aum_from_logits(logits, np.array([0, 2]))[0] > aum_from_logits(logits, np.array([0, 2]))[1]


# ---- порогові множини -----------------------------------------------------------------------
def test_threshold_sets_are_disjoint_and_sized_n_over_k_plus_1():
    sets = pick_threshold_sets(800, 4, seed=1)
    assert len(sets) == 2 and all(len(s) == 800 // 5 for s in sets)
    assert not set(sets[0]) & set(sets[1])
    assert sets[0].tolist() == pick_threshold_sets(800, 4, seed=1)[0].tolist()       # відтворюваність


# ---- compute_aum ----------------------------------------------------------------------------------
@pytest.fixture
def toy_aum(corpus_df):
    cfg = ExperimentConfig(epochs=3, hidden=32, batch_size=16, max_features=2000, seed=5)
    df = corpus_df.iloc[:150]
    labels = sorted(df.label.unique())
    y = df.label.map({l: i for i, l in enumerate(labels)}).to_numpy()
    return cfg, df.text.tolist(), y, len(labels)


def test_every_sample_scored_once_as_non_threshold(toy_aum):
    cfg, texts, y, k = toy_aum
    res = compute_aum(cfg, texts, y, k, 99.0, log=lambda *_: None)
    assert not np.isnan(res.aum).any() and set(res.run_of.tolist()) == {0, 1}
    for r, ids in enumerate(res.threshold_ids):
        assert (res.run_of[ids] != r).all()                    # порогові зразки прогону r оцінені не в ньому
    assert len(res.thresholds) == 2 and res.flags.shape == y.shape
    assert np.allclose(res.score, -res.aum)


def test_fit_uses_fake_class_and_threshold_labels(toy_aum, monkeypatch):
    cfg, texts, y, k = toy_aum
    calls = []
    real = aum_mod.fit_model

    def spy(cfg_, texts_, labels, num_labels, seed, epochs, **kw):
        calls.append({"labels": labels.copy(), "num_labels": num_labels, "seed": seed, "epochs": epochs,
                      "n": len(texts_), "logits": kw.get("track_logits")})
        return real(cfg_, texts_, labels, num_labels, seed, epochs, **kw)

    monkeypatch.setattr(aum_mod, "fit_model", spy)
    res = compute_aum(cfg, texts, y, k, 99.0, log=lambda *_: None)
    assert len(calls) == 2 and all(c["num_labels"] == k + 1 and c["n"] == len(y) and c["logits"] for c in calls)
    assert calls[0]["seed"] != calls[1]["seed"] and all(c["epochs"] == cfg.epochs for c in calls)
    for c, ids in zip(calls, res.threshold_ids):
        assert (c["labels"][ids] == k).all()                   # порогові зразки мають фіктивну мітку K+1
        assert (c["labels"][np.setdiff1d(np.arange(len(y)), ids)] < k).all()


def test_threshold_is_percentile_of_threshold_aum(toy_aum, monkeypatch):
    cfg, texts, y, k = toy_aum
    seen = []
    real = aum_mod.aum_from_logits

    def spy(logits, labels):
        a = real(logits, labels)
        seen.append((a, labels.copy()))
        return a

    monkeypatch.setattr(aum_mod, "aum_from_logits", spy)
    res = compute_aum(cfg, texts, y, k, 90.0, log=lambda *_: None)
    for (a, labels), thr in zip(seen, res.thresholds):
        assert thr == pytest.approx(np.percentile(a[labels == k], 90))
    assert (res.flags == (res.aum < np.asarray(res.thresholds)[res.run_of])).all()


def test_aum_skipped_when_too_few_samples(toy_aum):
    cfg, texts, y, k = toy_aum
    assert compute_aum(cfg, texts[:20], y[:20], k, 99.0, log=lambda *_: None) is None


def test_track_logits_shape_and_consistency_with_probabilities(toy_aum):
    cfg, texts, y, k = toy_aum
    f = fit_model(cfg, texts, y, k, 1, 3, track=True, track_logits=True)
    assert f.logits.shape == f.history.shape == (3, len(y), k)
    e = np.exp(f.logits - f.logits.max(axis=2, keepdims=True))
    assert np.allclose(e / e.sum(axis=2, keepdims=True), f.history, atol=1e-4)
    assert fit_model(cfg, texts, y, k, 1, 2).logits is None


def test_config_validation_for_aum():
    ExperimentConfig(aum_percentile=90).validate()
    for bad in (dict(aum_percentile=100), dict(aum_percentile=10)):
        with pytest.raises(AssertionError):
            ExperimentConfig(**bad).validate()


# ---- Dataset Cartography ------------------------------------------------------------------------
def test_cartography_formulas_match_naive_loops():
    rng = np.random.default_rng(2)
    hist = rng.dirichlet(np.ones(3), size=(4, 10))
    y = rng.integers(0, 3, 10)
    conf, var, corr = cartography(hist, y)
    for i in range(10):
        p = [hist[t, i, y[i]] for t in range(4)]
        assert conf[i] == pytest.approx(np.mean(p)) and var[i] == pytest.approx(np.std(p))
        assert corr[i] == pytest.approx(np.mean([hist[t, i].argmax() == y[i] for t in range(4)]))


def test_cartography_relates_to_C_and_D():
    from cleantext.scoring import risk_components
    rng = np.random.default_rng(3)
    hist = rng.dirichlet(np.ones(3), size=(5, 12))
    y = rng.integers(0, 3, 12)
    nb = np.array([[(i + 1 + j) % 12 for j in range(5)] for i in range(12)])
    c, d, _ = risk_components(hist, y, nb, 3)
    conf, _, corr = cartography(hist, y)
    assert np.allclose(conf, 1 - c) and np.allclose(corr, 1 - d)


# ---- конвеєр ----------------------------------------------------------------------------------------
@pytest.mark.slow
def test_pipeline_aum_artifacts(run_variants):
    out = run_variants["full"]
    det = pd.read_csv(out / "detection_metrics.csv")
    assert "aum" in set(det.key) and "margin" in set(det.key)
    cls = pd.read_csv(out / "classification_metrics.csv")
    row = cls[cls.key == "aum"].iloc[0]
    assert row.removed == row.K and row.n_train == cls[cls.key == "noisy"].n_train.iloc[0] - row.K
    a = np.load(out / "arrays.npz")
    assert np.allclose(a["score_aum"], -a["aum"]) and a["aum_flags"].dtype == bool
    assert {"cart_confidence", "cart_variability", "cart_correctness"} <= set(a.files)
    m = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["aum"]
    assert len(m["thresholds"]) == 2 and m["n_threshold"][0] == m["n_threshold"][1] and m["n_flagged"] == int(a["aum_flags"].sum())
    res = pd.read_csv(out / "review_candidates_research_only.csv")
    assert {"aum", "aum_flag", "confidence", "variability", "correctness"} <= set(res.columns)
    boot = pd.read_csv(out / "bootstrap_delta_ap.csv")
    assert "aum" in set(boot.versus)
    ex = pd.read_csv(out / "review_candidates_for_experts.csv")
    assert not {"aum", "confidence"} & set(ex.columns)


@pytest.mark.slow
def test_pipeline_without_aum(run_variants, tmp_path, monkeypatch, big_corpus_df):
    from cleantext import data as data_mod
    from cleantext import embeddings as emb_mod
    from cleantext.pipeline import run_experiment
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path / "cache")
    cfg = ExperimentConfig(dataset="csv", n_train=200, n_val=50, n_test=80, noise_rate=0.2, seed=3, embedding="tfidf_svd",
                           epochs=2, oof_folds=2, oof_epochs=2, bootstrap=5, k_neighbors=5, retrain_budgets=[0.10],
                           budget_frac=0.10, calibrate_weights=False, run_cleanlab=False, run_oracle=False, run_aum=False)
    out = run_experiment(cfg, log=lambda *_: None, csv_df=big_corpus_df, out_dir=tmp_path / "r")
    assert "aum" not in set(pd.read_csv(out / "detection_metrics.csv").key)
    assert json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["aum"] is None
    assert "aum" not in np.load(out / "arrays.npz").files
