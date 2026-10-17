import numpy as np
import pytest

from cleantext.noise import add_noise

KINDS = ["symmetric", "asymmetric", "semantic"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rate", [0.1, 0.2, 0.4])
def test_count_and_indicator(small_labels, ring_neighbours, kind, rate):
    y, z, _ = add_noise(small_labels, kind, rate, 1, 4, ring_neighbours)
    assert z.sum() == round(len(small_labels) * rate)
    assert np.array_equal(z, (y != small_labels).astype(int))


@pytest.mark.parametrize("kind", KINDS)
def test_zero_rate_changes_nothing(small_labels, ring_neighbours, kind):
    y, z, tr = add_noise(small_labels, kind, 0.0, 1, 4, ring_neighbours)
    assert np.array_equal(y, small_labels) and z.sum() == 0
    assert tr.sum() == tr.trace()


def test_asymmetric_shifts_class(small_labels):
    y, z, _ = add_noise(small_labels, "asymmetric", 0.3, 5, 4)
    idx = z == 1
    assert np.array_equal(y[idx], (small_labels[idx] + 1) % 4)


def test_semantic_takes_neighbour_label(small_labels, ring_neighbours):
    y, z, _ = add_noise(small_labels, "semantic", 0.3, 5, 4, ring_neighbours)
    for i in np.where(z == 1)[0]:
        first_other = next(j for j in ring_neighbours[i] if small_labels[j] != small_labels[i])
        assert y[i] == small_labels[first_other]


def test_symmetric_never_keeps_label_and_in_range(small_labels):
    y, z, _ = add_noise(small_labels, "symmetric", 0.5, 3, 4)
    assert (y[z == 1] != small_labels[z == 1]).all()
    assert y.min() >= 0 and y.max() < 4


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic_by_seed(small_labels, ring_neighbours, kind):
    a = add_noise(small_labels, kind, 0.2, 9, 4, ring_neighbours)[0]
    b = add_noise(small_labels, kind, 0.2, 9, 4, ring_neighbours)[0]
    c = add_noise(small_labels, kind, 0.2, 10, 4, ring_neighbours)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("kind", KINDS)
def test_transition_matrix_consistent(small_labels, ring_neighbours, kind):
    y, z, tr = add_noise(small_labels, kind, 0.2, 2, 4, ring_neighbours)
    assert tr.sum() == len(y)
    assert tr.sum() - tr.trace() == z.sum()
    for a in range(4):
        for b in range(4):
            assert tr[a, b] == ((small_labels == a) & (y == b)).sum()


def test_unknown_kind_raises(small_labels):
    with pytest.raises(ValueError):
        add_noise(small_labels, "bogus", 0.2, 1, 4)
