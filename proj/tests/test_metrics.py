import numpy as np
import pytest

from cleantext.metrics import (bootstrap_ap_diff, bootstrap_f1_diff, clf_metrics, detection_metrics,
                               precision_at_curve, top_k)

Z = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])


def test_perfect_ranking():
    s = np.array([.9, .8, .1, .2, .3, .05, .04, .03, .02, .01])
    m = detection_metrics(Z, s, 2)
    assert m["AP"] == pytest.approx(1.0) and m["ROC-AUC"] == pytest.approx(1.0)
    assert (m["Precision@K"], m["Recall@K"], m["TP"], m["FP"], m["FN"]) == (1.0, 1.0, 2, 0, 0)


def test_worst_ranking_counts():
    s = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    m = detection_metrics(Z, s, 2)
    assert (m["TP"], m["FP"], m["FN"]) == (0, 2, 2)
    assert m["Precision@K"] == 0 and m["Recall@K"] == 0
    assert m["ROC-AUC"] == pytest.approx(0.0)


def test_partial_hit():
    s = np.array([.9, .1, .8, .2, 0, 0, 0, 0, 0, 0])
    m = detection_metrics(Z, s, 2)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 1, 1)
    assert m["Precision@K"] == 0.5 and m["Recall@K"] == 0.5 and m["F1@K"] == pytest.approx(0.5)


def test_nan_without_positives():
    m = detection_metrics(np.zeros(10, dtype=int), np.arange(10.0), 3)
    assert np.isnan(m["AP"]) and np.isnan(m["ROC-AUC"]) and np.isnan(m["Recall@K"])
    assert m["TP"] == 0


def test_top_k_order_and_stability():
    assert top_k([0.1, 0.5, 0.5, 0.2], 2).tolist() == [1, 2]


def test_precision_at_curve():
    s = np.array([.9, .8, .1, .2, .3, .05, .04, .03, .02, .01])
    rows = precision_at_curve(Z, s, [0.2, 0.5])
    assert rows[0]["K"] == 2 and rows[0]["Precision@K"] == 1.0
    assert rows[1]["K"] == 5 and rows[1]["Recall@K"] == 1.0


def test_bootstrap_ap_deterministic_and_ci():
    rng = np.random.default_rng(0)
    z = (rng.random(200) < 0.2).astype(int)
    good = z + rng.normal(0, 0.4, 200)
    bad = rng.random(200)
    a = bootstrap_ap_diff(z, good, bad, 100, seed=4)
    b = bootstrap_ap_diff(z, good, bad, 100, seed=4)
    assert a == b
    point, lo, hi = a
    assert lo <= point <= hi and point > 0


def test_bootstrap_ap_nan_without_positives():
    res = bootstrap_ap_diff(np.zeros(20, dtype=int), np.arange(20.), np.arange(20.), 10)
    assert all(np.isnan(v) for v in res)


def test_clf_metrics():
    probs = np.array([[.9, .1], [.2, .8], [.6, .4], [.3, .7]])
    m = clf_metrics(np.array([0, 1, 1, 1]), probs)
    assert m["accuracy"] == pytest.approx(0.75)
    assert 0 < m["macro_f1"] < 1


def test_bootstrap_f1_diff_identical_predictions_zero():
    y = np.random.default_rng(0).integers(0, 3, 100)
    point, lo, hi = bootstrap_f1_diff(y, y, y, 50, seed=1)
    assert point == 0 == lo == hi
