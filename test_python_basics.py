import pytest

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)
from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Programming", 3),
        ("BCDFG", 0),
        ("queueing", 5),
        ("AbEcIdOfU", 5),
        ("a", 1),
        ("a" * 1000, 1000),
        ("", 0),
    ],
)
def test_count_vowels(text, expected):
    assert count_vowels(TextInput(text)) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("abcdefg", True),
        ("hello", False),
        ("abcABC", True),
        ("x", True),
        ("", True),
        ("abcdefga", False),
        ("zz", False),
    ],
)
def test_has_unique_characters(text, expected):
    assert has_unique_characters(TextInput(text)) is expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, 1),
        (5, 2),
        (8, 1),
        (15, 4),
        (1000, 6),
        (2 ** 20 - 1, 20),
        (2 ** 64, 1),
    ],
)
def test_count_one_bits(number, expected):
    assert count_one_bits(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, 0),
        (7, 0),
        (99, 2),
        (88, 3),
        (77, 4),
        (679, 5),
        (100, 1),
    ],
)
def test_multiplicative_persistence(number, expected):
    assert multiplicative_persistence(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    "predicted, expected, result",
    [
        ([2, 4], [2, 4], 0.0),
        ([1, 2, 3], [4, 5, 6], 9.0),
        ([0], [5], 25.0),
        ([-1, 1], [1, -1], 4.0),
        ([10, 20, 30, 40], [10, 20, 30, 44], 4.0),
    ],
)
def test_mse(predicted, expected, result):
    assert mse(VectorPairInput(predicted, expected)) == result


def test_mse_numpy_arrays():
    import numpy as np

    value = mse(VectorPairInput(np.array([1, 3]), np.array([2, 5])))
    assert value == 2.5


def test_mse_different_lengths():
    with pytest.raises(ValueError):
        mse(VectorPairInput([1, 2, 3], [1, 2]))


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, ""),
        (13, "(13)"),
        (30, "(2)(3)(5)"),
        (96, "(2**5)(3)"),
        (360, "(2**3)(3**2)(5)"),
        (1000, "(2**3)(5**3)"),
        (10403, "(101)(103)"),
    ],
)
def test_prime_factorization(number, expected):
    assert prime_factorization(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, 1),
        (55, 5),
        (91, 6),
        (204, 8),
        (3, "It is impossible"),
        (6, "It is impossible"),
        (100, "It is impossible"),
    ],
)
def test_pyramid(number, expected):
    assert pyramid(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (5, True),
        (22, True),
        (99, True),
        (1001, True),
        (12321, True),
        (3210123, True),
        (1000, False),
        (1203, False),
        (9876545, False),
    ],
)
def test_is_balanced_number(number, expected):
    assert is_balanced_number(PositiveIntegerInput(number)) is expected
