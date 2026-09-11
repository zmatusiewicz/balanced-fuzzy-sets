import pytest

from balanced_fuzzy_negations import (
    balanced_reversal,
    cosine,
    cube,
    reversal,
    square,
    square_root,
)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (-1.0, 1.0),
        (-0.5, 0.5),
        (0.0, 0.0),
        (0.5, -0.5),
        (1.0, -1.0),
    ],
)
def test_balanced_reversal_values(x, expected):
    assert balanced_reversal(x) == pytest.approx(expected)


@pytest.mark.parametrize(
    "negation_factory",
    [
        reversal,
        square,
        cube,
        square_root,
        cosine,
    ],
)
def test_known_negations_map_into_balanced_interval(negation_factory):
    negation = negation_factory()

    points = [
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]

    for x in points:
        assert -1.0 <= negation(x) <= 1.0


@pytest.mark.parametrize(
    "negation_factory",
    [
        reversal,
        square,
        cube,
        square_root,
        cosine,
    ],
)
def test_known_negations_reverse_sign(negation_factory):
    negation = negation_factory()

    for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        assert x * negation(x) <= pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "negation_factory",
    [
        reversal,
        square,
        cube,
        square_root,
        cosine,
    ],
)
def test_known_negations_have_fixed_zero(negation_factory):
    negation = negation_factory()

    assert negation(0.0) == pytest.approx(0.0)


def test_negation_rejects_value_outside_domain():
    negation = reversal()

    with pytest.raises(ValueError):
        negation(1.01)

    with pytest.raises(ValueError):
        negation(-1.01)
