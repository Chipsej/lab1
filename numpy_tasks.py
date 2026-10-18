import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = [np.asarray(A, dtype=float) for A in matrices]
    vectors = [np.asarray(x, dtype=float).reshape(-1, 1) for x in vectors]
    if not matrices:
        return np.zeros((0, 1), dtype=float)
    n = matrices[0].shape[0]
    result = np.zeros((n, 1), dtype=float)
    for A, x in zip(matrices, vectors):
        result += A @ x
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    matrix = np.asarray(matrix)
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return []
    return [np.unique(matrix[:, j]).tolist() for j in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    np.random.seed(seed)
    matrix = np.random.normal(mean, std, size=(rows, columns))

    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i, j = np.indices((rows, columns))
    return np.where((i + j) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[...] = np.asarray(background_color, dtype=np.uint8)
    cy, cx = image_height // 2, image_width // 2
    top = cy - height // 2
    left = cx - width // 2
    bottom = top + height
    right = left + width
    image[
        max(top, 0):min(bottom, image_height),
        max(left, 0):min(right, image_width),
    ] = np.asarray(shape_color, dtype=np.uint8)
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[...] = np.asarray(background_color, dtype=np.uint8)
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        return image
    cy, cx = image_height // 2, image_width // 2
    yy, xx = np.ogrid[:image_height, :image_width]
    mask = ((xx - cx) / semi_axis_x) ** 2 + ((yy - cy) / semi_axis_y) ** 2 <= 1
    image[mask] = np.asarray(shape_color, dtype=np.uint8)
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    variance = float(x.var())
    std = float(x.std())
    local_maxima = np.where(
        (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    )[0] + 1
    local_minima = np.where(
        (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])
    )[0] + 1
    if 0 < window <= len(x):
        moving_average = np.convolve(x, np.ones(window) / window, mode="valid")
    else:
        moving_average = np.array([], dtype=float)
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    if labels.size and int(labels.max()) >= class_count:
        raise ValueError("class_count is too small for given labels")
    result = np.zeros((labels.size, class_count), dtype=int)
    if labels.size and class_count > 0:
        result[np.arange(labels.size), labels] = 1
    return result
