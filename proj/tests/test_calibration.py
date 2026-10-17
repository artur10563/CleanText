"""Крок 3: калібрувальна підвибірка й адаптивні ваги (відсутність витоку, детермінізм, інтеграція)."""
import inspect
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cleantext import calibration as cal_mod
from cleantext import data as data_mod
from cleantext import embeddings as emb_mod
from cleantext.calibration import (best_weights, calib_noise, calibrate_weights, simplex_grid)
from cleantext.config import ExperimentConfig
from cleantext.data import build_calib, load_corpus
from cleantext.pipeline import run_experiment
from cleantext.runs import load_run
from tests.conftest import make_corpus_df


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


@pytest.fixture(scope="module")
def big_df():
    return make_corpus_df(per_class=250)         # 1000 текстів: вистачає на train 200 + val 50 + test 80 + calib 200


def _cfg(**kw):
    base = dict(dataset="csv", n_train=200, n_val=50, n_test=80, n_calib=200, noise_type="semantic", noise_rate=0.2,
                seed=7, embedding="tfidf_svd", epochs=2, oof_folds=2, oof_epochs=2, bootstrap=10, k_neighbors=5,
                retrain_budgets=[0.10], budget_frac=0.10, coeff_grid_step=0.25)
    base.update(kw)
    return ExperimentConfig(**base)


# ---- чисті функції --------------------------------------------------------------
def test_simplex_grid_counts_and_sums():
    for step, n in [(0.25, 15), (0.1, 66), (0.05, 231)]:
        g = simplex_grid(step)
        assert len(g) == n
        assert all(abs(a + b + c - 1) < 1e-9 and min(a, b, c) >= 0 for a, b, c in g)


def test_best_weights_tie_break_prefers_small_gamma_then_beta():
    t = pd.DataFrame({"alpha": [0.5, 0.0, 0.25, 0.0], "beta": [0.0, 0.5, 0.25, 0.0],
                      "gamma": [0.5, 0.5, 0.5, 1.0], "AP": [0.7, 0.7, 0.7, 0.7]})
    b = best_weights(t)
    assert (b.alpha, b.beta, b.gamma) == (0.5, 0.0, 0.5)
    t2 = pd.DataFrame({"alpha": [0.2, 0.9], "beta": [0.2, 0.1], "gamma": [0.6, 0.0], "AP": [0.9, 0.5]})
    assert best_weights(t2).alpha == 0.2                  # максимум AP важливіший за тай-брейк


def test_mixed_noise_has_rate_and_all_kinds(small_labels, ring_neighbours):
    y, z, _ = calib_noise(small_labels, "mixed", "semantic", 0.3, 5, 4, ring_neighbours)
    assert 0.2 < z.mean() < 0.4
    assert ((y != small_labels).astype(int) == z).all()
    y2, z2, _ = calib_noise(small_labels, "mixed", "symmetric", 0.3, 5, 4, ring_neighbours)
    assert (y == y2).all() and (z == z2).all()           # режим mixed не залежить від noise_type


def test_known_type_noise_matches_add_noise(small_labels, ring_neighbours):
    from cleantext.noise import add_noise
    a = calib_noise(small_labels, "known_type", "asymmetric", 0.2, 3, 4, ring_neighbours)[0]
    b = add_noise(small_labels, "asymmetric", 0.2, 3, 4, ring_neighbours)[0]
    assert (a == b).all()


# ---- відсутність витоку ---------------------------------------------------------
def test_calibrate_weights_signature_has_no_train_or_z():
    params = set(inspect.signature(calibrate_weights).parameters)
    assert params == {"cfg", "calib", "num_labels", "log", "cache_dir", "progress"}


def test_calibration_independent_of_target_scenario(isolated_cache, big_df):
    cfg = _cfg()
    corpus = load_corpus(cfg, log=lambda *_: None, csv_df=big_df)
    quiet = lambda *_: None
    a = calibrate_weights(cfg, corpus.calib, corpus.num_labels, log=quiet)
    # інший seed, інший рівень шуму й інші ваги-пріори цільового сценарію не змінюють результат калібрування
    other = replace(cfg, seed=99, noise_rate=0.4)
    b = calibrate_weights(other, corpus.calib, corpus.num_labels, log=quiet)
    assert a.weights == b.weights and a.ap == b.ap
    pd.testing.assert_frame_equal(a.table, b.table)
    c = calibrate_weights(cfg, corpus.calib, corpus.num_labels, log=quiet)
    assert c.weights == a.weights and c.ap == a.ap          # детермінізм
    # тип шуму цільового сценарію впливає лише в режимі known_type
    m1 = calibrate_weights(replace(cfg, calib_mode="mixed", noise_type="symmetric"), corpus.calib,
                           corpus.num_labels, log=quiet)
    m2 = calibrate_weights(replace(cfg, calib_mode="mixed", noise_type="semantic"), corpus.calib,
                           corpus.num_labels, log=quiet)
    assert m1.weights == m2.weights and m1.ap == m2.ap
    assert a.table.AP.max() == pytest.approx(a.ap)


def test_calibration_cache_roundtrip(isolated_cache, big_df):
    cfg = _cfg()
    corpus = load_corpus(cfg, log=lambda *_: None, csv_df=big_df)
    d = isolated_cache / "calibration"
    first = calibrate_weights(cfg, corpus.calib, corpus.num_labels, log=lambda *_: None, cache_dir=d)
    assert not first.info["from_cache"] and len(list(d.glob("calib_*.json"))) == 1
    second = calibrate_weights(cfg, corpus.calib, corpus.num_labels, log=lambda *_: None, cache_dir=d)
    assert second.info["from_cache"] and second.weights == first.weights and second.ap == first.ap
    # інший тип шуму (known_type) -> інший ключ кешу
    calibrate_weights(replace(cfg, noise_type="symmetric"), corpus.calib, corpus.num_labels,
                      log=lambda *_: None, cache_dir=d)
    assert len(list(d.glob("calib_*.json"))) == 2


# ---- дані -----------------------------------------------------------------------
def test_calib_disjoint_from_other_splits(isolated_cache, big_df):
    corpus = load_corpus(_cfg(), log=lambda *_: None, csv_df=big_df)
    assert corpus.calib is not None and len(corpus.calib) == 200
    sets = {n: set(map(data_mod.norm_text, getattr(corpus, n).text)) for n in ("train", "val", "test", "calib")}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert sets[a].isdisjoint(sets[b]), (a, b)
    assert set(corpus.calib.label) <= set(corpus.labels)
    assert corpus.stats["sizes"]["calib"] == 200 and "calib" in corpus.stats["split_hashes"]


def test_calib_does_not_change_other_splits(isolated_cache, big_df):
    with_c = load_corpus(_cfg(), log=lambda *_: None, csv_df=big_df)
    without = load_corpus(_cfg(calibrate_weights=False), log=lambda *_: None, csv_df=big_df)
    assert without.calib is None
    for n in ("train", "val", "test"):
        assert getattr(with_c, n).text.tolist() == getattr(without, n).text.tolist()


def test_build_calib_too_small_pool_returns_none():
    pool = pd.DataFrame({"text": [f"т{i}" for i in range(30)], "label": ["a", "b"] * 15})
    assert build_calib(pool, [], ["a", "b"], 100, 1, log=lambda *_: None) is None
    assert build_calib(pool, [], ["a", "b"], 0, 1) is None


# ---- конвеєр --------------------------------------------------------------------
@pytest.mark.slow
def test_pipeline_has_adaptive_next_to_fixed(isolated_cache, big_df):
    out = run_experiment(_cfg(), log=lambda *_: None, csv_df=big_df, out_dir=isolated_cache / "run")
    for name in ("calibration_table.csv", "calibration.json"):
        assert (out / name).exists(), name
    cal = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert cal["alpha"] + cal["beta"] + cal["gamma"] == pytest.approx(1)
    assert cal["n_calib"] == 200 and cal["seed"] == 2026 and cal["mode"] == "known_type"
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["coeff_fixed"] == [0.45, 0.35, 0.20]
    assert manifest["coeff_adaptive"] == [cal["alpha"], cal["beta"], cal["gamma"]]

    det = pd.read_csv(out / "detection_metrics.csv")
    assert {"proposed", "adaptive"} <= set(det.key)
    cls = pd.read_csv(out / "classification_metrics.csv")
    assert {"proposed", "adaptive"} <= set(cls.key)
    boot = pd.read_csv(out / "bootstrap_delta_ap.csv")
    assert {"proposed", "adaptive"} == set(boot.reference)

    # адаптивний R збігається з перерахунком за збереженими C, D, S і підібраними вагами
    a = np.load(out / "arrays.npz")
    expected = cal["alpha"] * a["comp_C"] + cal["beta"] * a["comp_D"] + cal["gamma"] * a["comp_S"]
    assert np.allclose(a["score_adaptive"], expected, atol=1e-6)

    run = load_run(out)
    assert run.calibration["alpha"] == cal["alpha"] and len(run.calibration_table) > 0
    assert set(run.bootstrap.reference) == {"proposed"} and set(run.bootstrap_adaptive.reference) == {"adaptive"}


@pytest.mark.slow
def test_target_noise_does_not_change_calibrated_weights(isolated_cache, big_df):
    kw = dict(run_retrain=False, run_cleanlab=False)
    o1 = run_experiment(_cfg(noise_rate=0.1, seed=3, **kw), log=lambda *_: None, csv_df=big_df,
                        out_dir=isolated_cache / "r1")
    o2 = run_experiment(_cfg(noise_rate=0.3, seed=4, **kw), log=lambda *_: None, csv_df=big_df,
                        out_dir=isolated_cache / "r2")
    c1 = json.loads((o1 / "calibration.json").read_text(encoding="utf-8"))
    c2 = json.loads((o2 / "calibration.json").read_text(encoding="utf-8"))
    # seed впливає на розбиття корпусу, тож calib різний; для однакового seed ваги мають бути однаковими
    o3 = run_experiment(_cfg(noise_rate=0.3, seed=3, **kw), log=lambda *_: None, csv_df=big_df,
                        out_dir=isolated_cache / "r3")
    c3 = json.loads((o3 / "calibration.json").read_text(encoding="utf-8"))
    assert (c1["alpha"], c1["beta"], c1["gamma"], c1["ap_calib"]) == (c3["alpha"], c3["beta"], c3["gamma"], c3["ap_calib"])
    assert c2["n_calib"] == c1["n_calib"]


@pytest.mark.slow
def test_calibration_off_equals_step2_behaviour(isolated_cache, big_df):
    out = run_experiment(_cfg(calibrate_weights=False, run_retrain=False, run_cleanlab=False), log=lambda *_: None,
                         csv_df=big_df, out_dir=isolated_cache / "off")
    assert not (out / "calibration.json").exists()
    det = pd.read_csv(out / "detection_metrics.csv")
    assert "adaptive" not in set(det.key)
    run = load_run(out)
    assert run.calibration is None and run.manifest["coeff_adaptive"] is None


@pytest.mark.slow
def test_retrain_includes_adaptive_with_same_k(isolated_cache, big_df, monkeypatch):
    from cleantext import pipeline as pipe_mod
    out = run_experiment(_cfg(run_cleanlab=False), log=lambda *_: None, csv_df=big_df, out_dir=isolated_cache / "rt")
    cls = pd.read_csv(out / "classification_metrics.csv")
    g = cls[(cls.budget_frac == 0.10) & ~cls.key.isin(["oracle", "weighted"])]
    assert {"random", "loss", "proposed", "adaptive"} <= set(g.key)
    assert g.removed.nunique() == 1
    assert len(load_run(out).removed_ids(0.10)["adaptive"]) == g.removed.iloc[0]


# ---- відповідність ноутбуку v2 -------------------------------------------------------
def test_calibration_model_seed_is_calib_seed_plus_999(isolated_cache, big_df, monkeypatch):
    seen = []
    real = cal_mod.fit_model

    def spy(cfg, texts, y, num_labels, seed, epochs, **kw):
        seen.append(seed)
        return real(cfg, texts, y, num_labels, seed, epochs, **kw)

    monkeypatch.setattr(cal_mod, "fit_model", spy)
    cfg = _cfg()
    corpus = load_corpus(cfg, log=lambda *_: None, csv_df=big_df)
    calibrate_weights(cfg, corpus.calib, corpus.num_labels, log=lambda *_: None)
    assert seen == [cfg.calib_seed + 999]


def test_fixed_point_is_a_candidate(isolated_cache, big_df):
    cfg = _cfg(alpha=0.45, beta=0.35, gamma=0.20)           # не вузол сітки з кроком 0,25
    corpus = load_corpus(cfg, log=lambda *_: None, csv_df=big_df)
    res = calibrate_weights(cfg, corpus.calib, corpus.num_labels, log=lambda *_: None)
    t = res.table
    row = t[(t.alpha == 0.45) & (t.beta == 0.35) & (t.gamma == 0.2)]
    assert len(row) == 1 and row.AP.iloc[0] == pytest.approx(res.fixed_ap)
    assert len(t) == 15 + 1 and res.ap >= res.fixed_ap - 1e-12
    assert res.ap == pytest.approx(t.AP.max())
