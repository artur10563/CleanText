import inspect

import numpy as np
import pytest

from cleantext import scoring
from cleantext.scoring import all_scores, combine, margin_score, risk_components

# T=2, N=3, K=2
HIST = np.array([
    [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]],
    [[0.8, 0.2], [0.4, 0.6], [0.3, 0.7]],
])
Y_OBS = np.array([0, 0, 0])
NEIGH = np.array([[1, 2], [0, 2], [0, 1]])


def test_components_hand_computed():
    c, d, s = risk_components(HIST, Y_OBS, NEIGH, k=2)
    assert c == pytest.approx([1 - (0.9 + 0.8) / 2, 1 - (0.2 + 0.4) / 2, 1 - (0.6 + 0.3) / 2])
    assert d == pytest.approx([0.0, 1.0, 0.5])      # argmax: [0,0] [1,1] [0,1]
    assert s == pytest.approx([0.0, 0.0, 0.0])      # усі спостережувані мітки однакові


def test_s_counts_disagreeing_neighbours():
    y = np.array([0, 1, 1])
    _, _, s = risk_components(HIST, y, NEIGH, k=2)
    assert s == pytest.approx([1.0, 0.5, 0.5])


def test_ranges():
    rng = np.random.default_rng(0)
    h = rng.dirichlet(np.ones(3), size=(5, 40))
    y = rng.integers(0, 3, 40)
    nb = rng.integers(0, 40, (40, 10))
    for comp in risk_components(h, y, nb, 10):
        assert comp.min() >= 0 and comp.max() <= 1


def test_combine_and_all_scores():
    c, d, s = risk_components(HIST, Y_OBS, NEIGH, 2)
    sc, comps = all_scores(HIST, Y_OBS, NEIGH, 2, 0.45, 0.35, 0.20, seed=1)
    assert sc["proposed"] == pytest.approx(0.45 * c + 0.35 * d + 0.20 * s)
    assert sc["proposed"] == pytest.approx(combine(c, d, s, 0.45, 0.35, 0.20))
    assert sc["dynamics"] == pytest.approx((0.45 * c + 0.35 * d) / 0.80)
    assert sc["semantic"] == pytest.approx(s)
    assert "cleanlab" not in sc and set(comps) == {"C", "D", "S"}


def test_loss_is_last_epoch_nll():
    sc, _ = all_scores(HIST, Y_OBS, NEIGH, 2, 0.45, 0.35, 0.20, seed=1)
    assert sc["loss"] == pytest.approx(-np.log(HIST[-1, np.arange(3), Y_OBS]))


def test_margin_sign_higher_when_suspicious():
    m = margin_score(HIST, Y_OBS)
    assert m[1] > m[0]  # зразок 1 «вважає» клас 1, спостережувана мітка 0


def test_cleanlab_added_with_oof():
    oof = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    sc, _ = all_scores(HIST, Y_OBS, NEIGH, 2, 0.45, 0.35, 0.20, seed=1, oof=oof)
    assert sc["cleanlab"] == pytest.approx([0.1, 0.9, 0.5])


def test_random_deterministic():
    a = all_scores(HIST, Y_OBS, NEIGH, 2, 0.45, 0.35, 0.20, seed=1)[0]["random"]
    b = all_scores(HIST, Y_OBS, NEIGH, 2, 0.45, 0.35, 0.20, seed=1)[0]["random"]
    assert np.array_equal(a, b)


# ---- відсутність витоку ---------------------------------------------------------
def test_signature_has_no_ground_truth():
    for fn in (all_scores, risk_components, scoring.per_epoch_scores):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"z", "clean", "clean_y", "y0", "y_true", "truth"}, (fn.__name__, names)


def test_scores_depend_only_on_observed_inputs():
    """Ризик — функція (історія, ỹ, сусіди): копії входів дають ідентичний результат."""
    rng = np.random.default_rng(3)
    h = rng.dirichlet(np.ones(3), size=(4, 30))
    y = rng.integers(0, 3, 30)
    nb = rng.integers(0, 30, (30, 10))
    first = all_scores(h, y, nb, 5, 0.45, 0.35, 0.20, seed=1)[0]
    second = all_scores(h.copy(), y.copy(), nb.copy(), 5, 0.45, 0.35, 0.20, seed=1)[0]
    for k in first:
        assert np.array_equal(first[k], second[k])
