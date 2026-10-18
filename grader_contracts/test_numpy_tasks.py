import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)

RED = (255, 0, 0)
BLACK = (0, 0, 0)


# ---------- sum_prod ----------

def test_sum_prod_identity_matrices():
    eye = np.eye(2)
    result = sum_prod(MatrixVectorBatchInput([eye, eye], [[1, 2], [3, 4]]))
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, [[4], [6]])


def test_sum_prod_single_matrix():
    result = sum_prod(MatrixVectorBatchInput([[[1, 2], [3, 4]]], [[1, 1]]))
    np.testing.assert_allclose(result, [[3], [7]])


def test_sum_prod_column_vectors():
    matrices = [np.array([[0, 1], [1, 0]]), np.array([[2, 0], [0, 3]])]
    vectors = [np.array([[1], [5]]), np.array([[1], [1]])]
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_allclose(result, [[7], [4]])


# ---------- binarize ----------

def test_binarize_threshold_is_strict():
    result = binarize(BinarizeInput(np.array([[1, 2, 3]]), threshold=2))
    np.testing.assert_array_equal(result, [[0, 0, 1]])


def test_binarize_negative_threshold_and_shape():
    m = np.array([[-5.0, -1.0], [0.0, 2.5], [-1.0, -0.5]])
    result = binarize(BinarizeInput(m, threshold=-1.0))
    assert result.shape == m.shape
    np.testing.assert_array_equal(result, [[0, 0], [1, 1], [0, 1]])


def test_binarize_returns_new_array():
    m = np.array([[5, 6], [7, 8]])
    result = binarize(BinarizeInput(m, threshold=6))
    np.testing.assert_array_equal(m, [[5, 6], [7, 8]])
    assert result is not m


# ---------- unique_rows / unique_columns ----------

def test_unique_rows():
    m = np.array([[3, 1, 3], [2, 2, 2], [1, 2, 3]])
    assert unique_rows(MatrixInput(m)) == [[1, 3], [2], [1, 2, 3]]


def test_unique_columns():
    m = np.array([[3, 1, 3], [2, 2, 2], [1, 2, 3]])
    assert unique_columns(MatrixInput(m)) == [[1, 2, 3], [1, 2], [2, 3]]


def test_unique_single_row_matrix():
    m = np.array([[4, 4, 5]])
    assert unique_rows(MatrixInput(m)) == [[4, 5]]
    assert unique_columns(MatrixInput(m)) == [[4], [4], [5]]


# ---------- matrix_statistics ----------

def test_matrix_statistics_same_seed_is_reproducible():
    data = RandomMatrixInput(rows=4, columns=6, mean=1.0, std=2.0, seed=42)
    a = matrix_statistics(data)
    b = matrix_statistics(data)
    np.testing.assert_array_equal(a.matrix, b.matrix)


def test_matrix_statistics_different_seeds_differ():
    a = matrix_statistics(RandomMatrixInput(rows=4, columns=6, mean=0.0, std=1.0, seed=1))
    b = matrix_statistics(RandomMatrixInput(rows=4, columns=6, mean=0.0, std=1.0, seed=2))
    assert not np.array_equal(a.matrix, b.matrix)


def test_matrix_statistics_match_numpy():
    s = matrix_statistics(RandomMatrixInput(rows=5, columns=3, mean=2.0, std=0.5, seed=7))
    assert s.matrix.shape == (5, 3)
    np.testing.assert_allclose(s.row_means, s.matrix.mean(axis=1))
    np.testing.assert_allclose(s.column_means, s.matrix.mean(axis=0))
    np.testing.assert_allclose(s.row_variances, s.matrix.var(axis=1))
    np.testing.assert_allclose(s.column_variances, s.matrix.var(axis=0))


def test_matrix_statistics_large_sample_distribution():
    s = matrix_statistics(RandomMatrixInput(rows=200, columns=200, mean=5.0, std=2.0, seed=0))
    assert abs(s.matrix.mean() - 5.0) < 0.1
    assert abs(s.matrix.std() - 2.0) < 0.1


# ---------- chess ----------

def test_chess_single_cell():
    result = chess(ChessInput(rows=1, columns=1, first=5, second=9))
    np.testing.assert_array_equal(result, [[5]])


def test_chess_custom_values():
    result = chess(ChessInput(rows=2, columns=2, first=7, second=9))
    np.testing.assert_array_equal(result, [[7, 9], [9, 7]])


def test_chess_neighbors_differ():
    result = chess(ChessInput(rows=5, columns=4, first=1, second=2))
    assert result.shape == (5, 4)
    assert result[0, 0] == 1
    assert np.all(result[:, 1:] != result[:, :-1])
    assert np.all(result[1:, :] != result[:-1, :])


# ---------- draw_rectangle ----------

def test_draw_rectangle_pixel_count_and_corner():
    img = draw_rectangle(RectangleInput(
        width=4, height=2, image_height=6, image_width=6,
        shape_color=RED, background_color=BLACK,
    ))
    assert img.shape == (6, 6, 3)
    red = np.all(img == RED, axis=-1)
    assert red.sum() == 8
    assert tuple(img[0, 0]) == BLACK


def test_draw_rectangle_centered_odd_sizes():
    img = draw_rectangle(RectangleInput(
        width=3, height=3, image_height=5, image_width=5,
        shape_color=RED, background_color=BLACK,
    ))
    red = np.all(img == RED, axis=-1)
    assert red[1:4, 1:4].all()
    assert red.sum() == 9


def test_draw_rectangle_fills_whole_image():
    img = draw_rectangle(RectangleInput(
        width=4, height=3, image_height=3, image_width=4,
        shape_color=RED, background_color=BLACK,
    ))
    assert img.shape == (3, 4, 3)
    assert np.all(img == RED)


# ---------- draw_ellipse ----------

def test_draw_ellipse_circle_is_symmetric():
    img = draw_ellipse(EllipseInput(
        semi_axis_x=3, semi_axis_y=3, image_height=7, image_width=7,
        shape_color=RED, background_color=BLACK,
    ))
    assert img.shape == (7, 7, 3)
    assert tuple(img[3, 3]) == RED
    assert tuple(img[0, 0]) == BLACK
    np.testing.assert_array_equal(img, img[::-1])
    np.testing.assert_array_equal(img, img[:, ::-1])


def test_draw_ellipse_non_square_image():
    img = draw_ellipse(EllipseInput(
        semi_axis_x=4, semi_axis_y=2, image_height=5, image_width=9,
        shape_color=RED, background_color=BLACK,
    ))
    assert img.shape == (5, 9, 3)
    assert tuple(img[2, 4]) == RED
    assert tuple(img[2, 0]) == RED
    assert tuple(img[0, 4]) == RED
    assert tuple(img[0, 0]) == BLACK


def test_draw_ellipse_huge_axes_cover_image():
    img = draw_ellipse(EllipseInput(
        semi_axis_x=100, semi_axis_y=100, image_height=4, image_width=6,
        shape_color=RED, background_color=BLACK,
    ))
    assert np.all(img == RED)


# ---------- analyze_time_series ----------

def test_time_series_full_example():
    s = analyze_time_series(TimeSeriesInput([5, 1, 4, 2, 3, 0], window=3))
    np.testing.assert_allclose(s.mean, 2.5)
    np.testing.assert_allclose(s.variance, 17.5 / 6)
    np.testing.assert_allclose(s.std, np.sqrt(17.5 / 6))
    np.testing.assert_array_equal(s.local_maxima_indices, [2, 4])
    np.testing.assert_array_equal(s.local_minima_indices, [1, 3])
    np.testing.assert_allclose(s.moving_average, [10 / 3, 7 / 3, 3.0, 5 / 3])


def test_time_series_constant():
    s = analyze_time_series(TimeSeriesInput([2, 2, 2, 2], window=2))
    assert s.variance == 0.0
    assert len(s.local_maxima_indices) == 0
    assert len(s.local_minima_indices) == 0


def test_time_series_plateau_is_not_extremum():
    s = analyze_time_series(TimeSeriesInput([1, 3, 3, 1], window=1))
    assert len(s.local_maxima_indices) == 0
    assert len(s.local_minima_indices) == 0


def test_time_series_monotonic_has_no_extrema():
    s = analyze_time_series(TimeSeriesInput([1, 2, 3, 4, 5], window=2))
    assert len(s.local_maxima_indices) == 0
    assert len(s.local_minima_indices) == 0


def test_time_series_window_edge_cases():
    values = [4, 8, 6, 2]
    one = analyze_time_series(TimeSeriesInput(values, window=1))
    np.testing.assert_allclose(one.moving_average, values)
    full = analyze_time_series(TimeSeriesInput(values, window=4))
    np.testing.assert_allclose(full.moving_average, [5.0])
    mid = analyze_time_series(TimeSeriesInput(values, window=3))
    assert len(mid.moving_average) == len(values) - 3 + 1


# ---------- one_hot ----------

def test_one_hot_basic():
    result = one_hot(OneHotInput([1, 0, 1]))
    np.testing.assert_array_equal(result, [[0, 1], [1, 0], [0, 1]])


def test_one_hot_single_label():
    result = one_hot(OneHotInput([0]))
    np.testing.assert_array_equal(result, [[1]])


def test_one_hot_explicit_class_count():
    result = one_hot(OneHotInput([0, 1], class_count=5))
    assert result.shape == (2, 5)
    np.testing.assert_array_equal(result, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])


def test_one_hot_rows_sum_to_one():
    result = one_hot(OneHotInput([3, 0, 2, 2, 1]))
    assert result.shape == (5, 4)
    np.testing.assert_array_equal(result.sum(axis=1), np.ones(5))


def test_one_hot_class_count_too_small():
    with pytest.raises(ValueError):
        one_hot(OneHotInput([0, 3], class_count=3))
