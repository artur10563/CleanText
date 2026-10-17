"""Еталонні перевірки: правила docx (методичний план) і ноутбука v2, виписані окремо від реалізації.

Формули обчислюються наївними циклами за буквою docx 4.2–4.4, а не викликом функцій реалізації.
"""
import json

import numpy as np
import pandas as pd
import pytest

from cleantext.config import BERT_PRESETS, PRESETS, ExperimentConfig
from cleantext.embeddings import nearest_neighbours
from cleantext.metrics import detection_metrics, top_k
from cleantext.scoring import all_scores, combine, margin_score, risk_components


# ---- docx 4.2: формули ознак ---------------------------------------------------------
@pytest.fixture
def toy():
    rng = np.random.default_rng(3)
    T, n, K = 4, 12, 3
    hist = rng.dirichlet(np.ones(K), size=(T, n)).astype(np.float64)
    y = rng.integers(0, K, n)
    nb = np.array([[(i + 1 + j) % n for j in range(5)] for i in range(n)])
    return hist, y, nb


def test_C_D_S_match_docx_formulas(toy):
    hist, y, nb = toy
    T, n, _ = hist.shape
    k = 4
    c, d, s = risk_components(hist, y, nb, k)
    for i in range(n):
        c_i = 1 - sum(hist[t, i, y[i]] for t in range(T)) / T                          # C_i
        d_i = 1 - sum(int(hist[t, i].argmax() == y[i]) for t in range(T)) / T          # D_i
        s_i = 1 - sum(int(y[j] == y[i]) for j in nb[i][:k]) / k                        # S_i
        assert c[i] == pytest.approx(c_i) and d[i] == pytest.approx(d_i) and s[i] == pytest.approx(s_i)


def test_R_is_convex_combination_in_unit_interval(toy):
    hist, y, nb = toy
    c, d, s = risk_components(hist, y, nb, 4)
    r = combine(c, d, s, 0.45, 0.35, 0.20)
    assert np.allclose(r, 0.45 * c + 0.35 * d + 0.20 * s)
    assert r.min() >= 0 and r.max() <= 1


def test_neighbours_exclude_the_sample_itself():
    emb = np.random.default_rng(0).normal(size=(40, 8)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    nb = nearest_neighbours(emb, 10)
    assert all(i not in nb[i] for i in range(len(emb)))
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    assert (nb[:, 0] == sims.argmax(axis=1)).all()            # перший сусід справді найближчий за косинусом


def test_baselines_loss_last_epoch_and_margin_formula(toy):
    hist, y, nb = toy
    scores, _ = all_scores(hist, y, nb, 4, 0.45, 0.35, 0.20, seed=1)
    n = len(y)
    assert np.allclose(scores["loss"], [-np.log(hist[-1, i, y[i]]) for i in range(n)])      # лише остання епоха
    exp = [np.mean([max(v for c, v in enumerate(hist[t, i]) if c != y[i]) - hist[t, i, y[i]]
                    for t in range(hist.shape[0])]) for i in range(n)]
    assert np.allclose(margin_score(hist, y), exp)


def test_ablations_follow_docx_definitions(toy):
    hist, y, nb = toy
    scores, comps = all_scores(hist, y, nb, 4, 0.45, 0.35, 0.20, seed=1)
    assert np.allclose(scores["semantic"], comps["S"])                                      # R(0,0,S)
    assert np.allclose(scores["dynamics"], (0.45 * comps["C"] + 0.35 * comps["D"]) / 0.80)  # R(C,D,0), перенормовано
    assert np.allclose(scores["proposed"], combine(comps["C"], comps["D"], comps["S"], .45, .35, .2))


def test_top_k_is_descending_risk_order():
    r = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k(r, 3).tolist() == [1, 3, 2]


def test_detection_metrics_definitions():
    """docx розд. 8: TP, FP, FN, Precision, Recall, F1."""
    z = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .1, .7, .6, .2, .3, .0])
    m = detection_metrics(z, s, 4)
    assert (m["TP"], m["FP"], m["FN"]) == (2, 2, 1)
    assert m["Precision@K"] == pytest.approx(2 / 4) and m["Recall@K"] == pytest.approx(2 / 3)
    assert m["F1@K"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert {"AP", "PR-AUC", "ROC-AUC"} <= set(m)


# ---- параметри за замовчуванням: docx і v2 --------------------------------------------
def test_defaults_match_docx_and_v2():
    c = ExperimentConfig()
    assert (c.alpha, c.beta, c.gamma) == (0.45, 0.35, 0.20)             # docx 4.3 / v2 FIXED_COEFF
    assert c.budget_frac == 0.20 and c.k_neighbors == 10                # docx: TOP K = 20 %; v2 TOP_K_NEIGHBORS
    assert (c.calib_noise_rate, c.calib_seed, c.coeff_grid_step) == (0.20, 2026, 0.25)   # v2
    assert c.oof_epochs == c.epochs                                      # v2 OOF_EPOCHS = EPOCHS
    assert PRESETS["quick"]["n_calib"] == 350 and PRESETS["main"]["n_calib"] == 700      # v2 N_CALIB
    assert PRESETS["main"]["retrain_budgets"] == [0.05, 0.10, 0.15, 0.20]                # v2 REVIEW_BUDGETS (повний режим)
    assert BERT_PRESETS["main"]["retrain_budgets"] == [0.05, 0.10, 0.15, 0.20]
    assert 0.20 in c.all_budgets() and 0.05 in c.all_budgets()           # docx: додатково K = 5/10/20 %


# ---- docx розд. 8: що зберігається ----------------------------------------------------
@pytest.mark.slow
def test_manifest_stores_required_reproducibility_fields(run_variants):
    out = run_variants["full"]
    m = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = m["config"]
    assert {"seed", "noise_type", "noise_rate", "alpha", "beta", "gamma", "epochs", "k_neighbors"} <= set(cfg)
    assert {"train", "val", "test", "calib"} <= set(m["data"]["split_hashes"])          # хеші розбиттів
    assert {"train", "val", "test", "calib"} <= set(m["data"]["sizes"])                 # кількість прикладів
    assert m["removed_ids"] and m["retrain_budgets"] and m["budget_K"]                  # ID, бюджети
    assert m["timings"] and "device" in m                                                # тривалість, GPU/CPU
    assert "torch" in m["versions"] and "sklearn" in m["versions"]                       # версії бібліотек
    tr = np.load(out / "arrays.npz")["transition"]
    off_diag = tr.sum() - tr.trace()                                                    # матриця переходів: змінені — поза діагоналлю
    assert tr.shape[0] == tr.shape[1] == len(m["labels"]) and off_diag == m["n_corrupted"]
    assert tr.sum() == m["data"]["sizes"]["train"]


@pytest.mark.xfail(reason="Ревізії HF-датасетів/моделей і параметри токенізації ще не зберігаються — крок 6",
                   strict=True)
@pytest.mark.slow
def test_manifest_stores_model_revisions(run_variants):
    m = json.loads((run_variants["full"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert "dataset_revision" in m["data"]["source"] and "tokenizer" in m["config"]


@pytest.mark.slow
def test_equal_removed_count_across_methods(run_variants):
    """docx розд. 8: однакова кількість вилучених текстів для методів (крім oracle при K > |z|)."""
    cls = pd.read_csv(run_variants["full"] / "classification_metrics.csv")
    g = cls[~cls.key.isin(["noisy", "oracle", "weighted"])]
    assert g.groupby("budget_frac").removed.nunique().eq(1).all()


@pytest.mark.slow
def test_class_balance_of_removed_is_reported(run_variants):
    rbc = pd.read_csv(run_variants["full"] / "removed_by_class.csv")
    assert {"budget_frac", "key", "class", "count"} <= set(rbc.columns) and len(rbc)


@pytest.mark.slow
def test_expert_register_hides_risk_components_and_truth(run_variants):
    """docx розд. 9: анотатор бачить текст і мітку, але не R-розкладання, не z, не мітку датасету."""
    ex = pd.read_csv(run_variants["full"] / "review_candidates_for_experts.csv")
    forbidden = {"synthetic_corrupted", "original_dataset_label", "C_confidence_risk", "D_disagreement_risk",
                 "S_semantic_risk", "cleanlab_risk", "cleanlab_flag"}
    assert not forbidden & set(ex.columns)


@pytest.mark.xfail(reason="Реєстр для експертів ще містить proposed_risk і rank (docx розд. 9: приховати Rᵢ) — крок 7",
                   strict=True)
@pytest.mark.slow
def test_expert_register_hides_R_and_rank(run_variants):
    ex = pd.read_csv(run_variants["full"] / "review_candidates_for_experts.csv")
    assert not {"proposed_risk", "rank"} & set(ex.columns)
