import numpy as np
import pytest

from neural_network_lab.losses import (
    mean_squared_error,
    mean_squared_error_derivative,
)


def test_mean_squared_error_returns_average_squared_difference():
    y_true = [1.0, 0.0, 1.0]
    y_pred = [0.7, 0.2, 0.9]
    expected = 0.14 / 3

    mse = mean_squared_error(y_true, y_pred)

    np.testing.assert_allclose(mse, expected, rtol=1e-7)


def test_mean_squared_error_derivative_returns_expected_gradient():
    y_true = [1.0, 0.0, 1.0]
    y_pred = [0.7, 0.2, 0.9]
    expected = np.array([-0.6, 0.4, -0.2]) / 3

    mse_derivative = mean_squared_error_derivative(y_true, y_pred)

    np.testing.assert_allclose(mse_derivative, expected, rtol=1e-7)


def test_mean_squared_error_raises_value_error_for_mismatched_shapes():
    y_true = [1.0, 2.0]
    y_pred = [1.0]

    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred)


def test_mean_squared_error_derivative_uses_total_number_of_elements():
    y_true = [[1.0, 0.0], [1.0, 0.0]]
    y_pred = [[0.5, 0.2], [0.8, 0.4]]
    expected = [[-0.25, 0.1], [-0.1, 0.2]]

    mse_derivative = mean_squared_error_derivative(y_true, y_pred)

    np.testing.assert_allclose(mse_derivative, expected, rtol=1e-7)


def test_mean_squared_error_derivative_raises_value_error_for_mismatched_shapes():
    y_true = [[1.0, 2.0], [3.0, 4.0]]
    y_pred = [[1.0, 2.0]]

    with pytest.raises(ValueError):
        mean_squared_error_derivative(y_true, y_pred)
