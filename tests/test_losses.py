import numpy as np
import pytest

from neural_network_lab.losses import (
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    categorical_cross_entropy,
    categorical_cross_entropy_derivative,
    softmax_categorical_cross_entropy_derivative,
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


def test_binary_cross_entropy_returns_average_log_loss():
    y_true = [1, 0]
    y_pred = [0.8, 0.25]
    expected = (-np.log(0.8) - np.log(0.75)) / 2

    bce = binary_cross_entropy(y_true, y_pred)

    np.testing.assert_allclose(bce, expected, rtol=1e-7)


def test_binary_cross_entropy_stability_for_extreme_probabilities():
    y_true = [1, 0]
    y_pred = [1.0, 0.0]

    with np.errstate(divide="raise", invalid="raise"):
        bce = binary_cross_entropy(y_true, y_pred)

    assert np.isfinite(bce)
    np.testing.assert_allclose(bce, 0.0, atol=1e-10)


def test_binary_cross_entropy_derivative_returns_expected_gradient():
    y_true = [1, 0]
    y_pred = [0.8, 0.25]
    expected = [-0.625, 2 / 3]

    bce_derivative = binary_cross_entropy_derivative(y_true, y_pred)

    np.testing.assert_allclose(bce_derivative, expected, rtol=1e-7)


def test_binary_cross_entropy_derivative_uses_total_number_of_elements():
    y_true = [[1, 0], [0, 1]]
    y_pred = [[0.8, 0.25], [0.1, 0.9]]
    expected = [[-0.3125, 1 / 3], [5 / 18, -5 / 18]]

    bce_derivative = binary_cross_entropy_derivative(y_true, y_pred)

    np.testing.assert_allclose(bce_derivative, expected, rtol=1e-7)


@pytest.mark.parametrize(
    "loss_function",
    [binary_cross_entropy, binary_cross_entropy_derivative],
)
def test_binary_cross_entropy_functions_reject_non_binary_targets(loss_function):
    y_true = [0.5]
    y_pred = [0.5]

    with pytest.raises(ValueError):
        loss_function(y_true, y_pred)


@pytest.mark.parametrize(
    "loss_function",
    [binary_cross_entropy, binary_cross_entropy_derivative],
)
def test_binary_cross_entropy_functions_reject_invalid_probabilities(loss_function):
    y_true = [1]
    y_pred = [1.2]

    with pytest.raises(ValueError):
        loss_function(y_true, y_pred)


@pytest.mark.parametrize(
    "loss_function",
    [binary_cross_entropy, binary_cross_entropy_derivative],
)
def test_binary_cross_entropy_functions_reject_mismatched_shapes(loss_function):
    y_true = [1, 0]
    y_pred = [0.8]

    with pytest.raises(ValueError):
        loss_function(y_true, y_pred)


def test_binary_cross_entropy_derivative_is_finite_at_probability_boundaries():
    y_true = [1, 0]
    y_pred = [0.0, 1.0]

    with np.errstate(divide="raise", invalid="raise"):
        bce_derivative = binary_cross_entropy_derivative(y_true, y_pred)

    assert np.all(np.isfinite(bce_derivative))


@pytest.mark.parametrize(
    "loss_function",
    [binary_cross_entropy, binary_cross_entropy_derivative],
)
def test_binary_cross_entropy_functions_reject_nan_probabilities(loss_function):
    y_true = [1]
    y_pred = [np.nan]

    with pytest.raises(ValueError):
        loss_function(y_true, y_pred)


def test_categorical_cross_entropy_returns_loss_for_single_sample():
    y_true = [0, 1, 0]
    y_pred = [0.1, 0.7, 0.2]
    expected = -np.log(0.7)

    cce = categorical_cross_entropy(y_true, y_pred)

    np.testing.assert_allclose(cce, expected, rtol=1e-7)


def test_categorical_cross_entropy_returns_average_loss_over_batch():
    y_true = [[0, 1, 0], [1, 0, 0]]
    y_pred = [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]
    expected = (-np.log(0.7) - np.log(0.8)) / 2

    cce = categorical_cross_entropy(y_true, y_pred)

    np.testing.assert_allclose(cce, expected, rtol=1e-7)


def test_categorical_cross_entropy_derivative_returns_expected_gradient():
    y_true = [0, 1, 0]
    y_pred = [0.1, 0.7, 0.2]
    expected = [-0.0, -1 / 0.7, -0.0]

    cce_derivative = categorical_cross_entropy_derivative(y_true, y_pred)

    np.testing.assert_allclose(cce_derivative, expected, rtol=1e-7)


def test_categorical_cross_entropy_derivative_uses_batch_size():
    y_true = [[0, 1, 0], [1, 0, 0]]
    y_pred = [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]
    expected = [
        [0.0, -1 / (2 * 0.7), 0.0],
        [-1 / (2 * 0.8), 0.0, 0.0],
    ]

    cce_derivative = categorical_cross_entropy_derivative(y_true, y_pred)

    np.testing.assert_allclose(cce_derivative, expected, rtol=1e-7)


def test_softmax_categorical_cross_entropy_derivative_returns_expected_gradient():
    y_true = [0, 1, 0]
    y_pred = [0.1, 0.7, 0.2]
    expected = [0.1, -0.3, 0.2]

    softmax_cce_derivative = softmax_categorical_cross_entropy_derivative(y_true, y_pred)

    np.testing.assert_allclose(softmax_cce_derivative, expected, rtol=1e-7)


def test_softmax_categorical_cross_entropy_derivative_uses_batch_size():
    y_true = [[0, 1, 0], [1, 0, 0]]
    y_pred = [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]
    expected = [
        [0.05, -0.15, 0.10],
        [-0.10, 0.05, 0.05],
    ]

    softmax_cce_derivative = softmax_categorical_cross_entropy_derivative(y_true, y_pred)

    np.testing.assert_allclose(softmax_cce_derivative, expected, rtol=1e-7)


def test_categorical_cross_entropy_is_stable_at_probability_boundaries():
    y_true = [0, 1, 0]
    y_pred = [0.0, 1.0, 0.0]

    with np.errstate(divide="raise", invalid="raise"):
        cce = categorical_cross_entropy(y_true, y_pred)

    assert np.isfinite(cce)
    np.testing.assert_allclose(cce, 0.0, atol=1e-10)


def test_categorical_cross_entropy_derivative_is_finite_at_probability_boundaries():
    y_true = [0, 1, 0]
    y_pred = [1.0, 0.0, 0.0]

    with np.errstate(divide="raise", invalid="raise"):
        cce_derivative = categorical_cross_entropy_derivative(y_true, y_pred)

    assert np.all(np.isfinite(cce_derivative))

@pytest.mark.parametrize(
    "loss_function",
    [
        categorical_cross_entropy,
        categorical_cross_entropy_derivative,
        softmax_categorical_cross_entropy_derivative,
    ],
)
@pytest.mark.parametrize(("y_true", "y_pred"), [
    # Mismatched shapes
    ([0, 1, 0], [0.2, 0.8]),

    # Unsupported three-dimensional inputs
    ([[[0, 1, 0]]], [[[0.1, 0.7, 0.2]]]),

    # Empty inputs
    ([], []),

    # Target is not one-hot
    ([1, 1, 0], [0.1, 0.7, 0.2]),

    # Probabilities do not sum to one
    ([0, 1, 0], [0.1, 0.7, 0.1]),

    # Probability outside the valid range, while still summing to one
    ([0, 1, 0], [-0.1, 0.7, 0.4]),

    # Non-finite probability
    ([0, 1, 0], [0.1, np.nan, 0.9]),
    ],
)
def test_categorical_cross_entropy_functions_reject_invalid_inputs(
    loss_function,
    y_true,
    y_pred
    ):
    with pytest.raises(ValueError):
        loss_function(y_true, y_pred)