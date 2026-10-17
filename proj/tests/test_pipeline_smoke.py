"""Наскрізний тест конвеєра на синтетичному корпусі (без мережі, tfidf_svd + MLP)."""
import json

import numpy as np
import pandas as pd
import pytest

from cleantext import data as data_mod
from cleantext import embeddings as emb_mod
from cleantext.config import ExperimentConfig
from cleantext.data import load_corpus
from cleantext.pipeline import run_experiment

EXPECTED_FILES = [
    "review_candidates_research_only.csv", "review_candidates_for_experts.csv", "learning_log.csv",
    "detection_metrics.csv", "classification_metrics.csv", "budget_curve.csv",
    "epoch_detection_metrics.csv", "bootstrap_delta_ap.csv", "arrays.npz", "oof_probabilities.npy",
    "run_manifest.json", "train.parquet", "val.parquet", "test.parquet",
]


def _cfg(**kw):
    base = dict(dataset="csv", n_train=200, n_val=50, n_test=80, noise_type="semantic", noise_rate=0.2,
                seed=7, embedding="tfidf_svd", classifier="mlp", epochs=2, oof_folds=2, oof_epochs=1,
                bootstrap=20, k_neighbors=5)
    base.update(kw)
    return ExperimentConfig(**base)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


@pytest.mark.slow
def test_end_to_end_artifacts_and_blindness(isolated_cache, corpus_df):
    cfg = _cfg()
    out = run_experiment(cfg, log=lambda *_: None, csv_df=corpus_df, out_dir=isolated_cache / "run")
    for name in EXPECTED_FILES:
        assert (out / name).exists(), name

    # сліпий реєстр: експерту не видно ні істинної мітки, ні індикатора шуму
    expert = pd.read_csv(out / "review_candidates_for_experts.csv")
    assert not {"synthetic_corrupted", "original_dataset_label"} & set(expert.columns)
    research = pd.read_csv(out / "review_candidates_research_only.csv")
    assert {"synthetic_corrupted", "original_dataset_label"} <= set(research.columns)

    # шум лише в train: val/test містять істинні мітки з корпусу
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    labels = manifest["labels"]
    lookup = dict(zip(corpus_df.text, corpus_df.label))
    for name in ("val", "test"):
        df = pd.read_parquet(out / f"{name}.parquet")
        assert [labels[i] for i in df.y] == [lookup[t] for t in df.text]
        assert (df.label == [lookup[t] for t in df.text]).all()

    arrays = np.load(out / "arrays.npz")
    assert int(arrays["z"].sum()) == manifest["n_corrupted"] == round(len(arrays["z"]) * 0.2)
    assert manifest["budget_K"] == round(len(arrays["z"]) * cfg.budget_frac)
    assert arrays["history"].shape == (cfg.epochs, len(arrays["z"]), len(labels))
    # OOF-ймовірності — розподіли
    assert np.allclose(arrays["oof"].sum(axis=1), 1, atol=1e-4)

    det = pd.read_csv(out / "detection_metrics.csv")
    assert {"random", "loss", "margin", "cleanlab", "proposed"} <= set(det.key)
    cls = pd.read_csv(out / "classification_metrics.csv")
    assert {"noisy", "random", "loss", "proposed", "cleanlab", "oracle"} <= set(cls.key)


@pytest.mark.slow
def test_zero_noise_run_has_nan_detection(isolated_cache, corpus_df):
    out = run_experiment(_cfg(noise_rate=0.0, run_retrain=False, run_cleanlab=False),
                         log=lambda *_: None, csv_df=corpus_df, out_dir=isolated_cache / "run0")
    det = pd.read_csv(out / "detection_metrics.csv")
    assert det["AP"].isna().all()


def test_corpus_splits_are_disjoint(isolated_cache, corpus_df):
    corpus = load_corpus(_cfg(), log=lambda *_: None, csv_df=corpus_df)
    sets = {n: set(map(data_mod.norm_text, getattr(corpus, n).text)) for n in ("train", "val", "test")}
    assert sets["train"].isdisjoint(sets["val"]) and sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    assert corpus.num_labels == 4
