import numpy as np
import pytest

from neural_network_lab.layers import Dense
from neural_network_lab.optimizers import Momentum, SGD


def test_sgd_updates_weights_and_biases():
    layer = Dense(2, 2, seed=42)
    layer.weights = np.array([
        [0.5, 1.0],
        [-0.5, 0.25],
    ])
    layer.biases = np.array([0.1, -0.2])

    layer.forward([2.0, -1.0])
    layer.backward([0.3, -0.2])

    optimizer = SGD(learning_rate=0.1)
    optimizer.step(layer)

    expected_weights = np.array([
        [0.44, 1.04],
        [-0.47, 0.23],
    ])
    expected_biases = np.array([0.07, -0.18])

    np.testing.assert_allclose(
        layer.weights,
        expected_weights,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        layer.biases,
        expected_biases,
        rtol=1e-7,
    )


def test_sgd_requires_gradients_before_updating_parameters():
    layer = Dense(2, 2, seed=42)
    optimizer = SGD(learning_rate=0.1)

    with pytest.raises(RuntimeError):
        optimizer.step(layer)


@pytest.mark.parametrize(
    "invalid_learning_rate",
    [
        0,
        -0.1,
        True,
        np.nan,
        np.inf,
        -np.inf,
        "0.1",
        None,
    ],
)
def test_sgd_rejects_invalid_learning_rates(
    invalid_learning_rate,
):
    with pytest.raises(ValueError):
        SGD(learning_rate=invalid_learning_rate)


def test_momentum_preserves_velocity_between_steps():
    layer = Dense(1, 1, seed=42)
    layer.weights.fill(1.0)
    layer.biases.fill(0.0)

    optimizer = Momentum(
        learning_rate=0.1,
        momentum=0.9,
    )

    layer.forward([1.0])
    layer.backward([0.5])
    optimizer.step(layer)

    np.testing.assert_allclose(layer.weights, [[0.95]])
    np.testing.assert_allclose(layer.biases, [-0.05])

    layer.forward([1.0])
    layer.backward([0.5])
    optimizer.step(layer)

    np.testing.assert_allclose(layer.weights, [[0.855]])
    np.testing.assert_allclose(layer.biases, [-0.145])


def test_momentum_keeps_separate_velocities_for_each_layer():
    first_layer = Dense(1, 1, seed=42)
    second_layer = Dense(1, 1, seed=42)

    for layer in (first_layer, second_layer):
        layer.weights.fill(1.0)
        layer.biases.fill(0.0)

    optimizer = Momentum(
        learning_rate=0.1,
        momentum=0.9,
    )

    first_layer.forward([1.0])
    first_layer.backward([0.5])
    optimizer.step(first_layer)

    second_layer.forward([1.0])
    second_layer.backward([1.0])
    optimizer.step(second_layer)

    np.testing.assert_allclose(first_layer.weights, [[0.95]])
    np.testing.assert_allclose(first_layer.biases, [-0.05])
    np.testing.assert_allclose(second_layer.weights, [[0.9]])
    np.testing.assert_allclose(second_layer.biases, [-0.1])

    first_layer.forward([1.0])
    first_layer.backward([0.5])
    optimizer.step(first_layer)

    np.testing.assert_allclose(first_layer.weights, [[0.855]])
    np.testing.assert_allclose(first_layer.biases, [-0.145])
    np.testing.assert_allclose(second_layer.weights, [[0.9]])
    np.testing.assert_allclose(second_layer.biases, [-0.1])


def test_momentum_zero_matches_sgd():
    sgd_layer = Dense(1, 1, seed=42)
    momentum_layer = Dense(1, 1, seed=42)

    sgd = SGD(learning_rate=0.1)
    momentum = Momentum(
        learning_rate=0.1,
        momentum=0.0,
    )

    for gradient in [0.5, -0.2, 0.1]:
        sgd_layer.forward([1.0])
        sgd_layer.backward([gradient])
        sgd.step(sgd_layer)

        momentum_layer.forward([1.0])
        momentum_layer.backward([gradient])
        momentum.step(momentum_layer)

        np.testing.assert_allclose(
            momentum_layer.weights,
            sgd_layer.weights,
        )
        np.testing.assert_allclose(
            momentum_layer.biases,
            sgd_layer.biases,
        )


@pytest.mark.parametrize(
    "invalid_momentum",
    [
        -0.1,
        1.0,
        1.1,
        True,
        np.nan,
        np.inf,
        -np.inf,
        "0.9",
        None,
    ],
)
def test_momentum_rejects_invalid_momentum(invalid_momentum):
    with pytest.raises(ValueError):
        Momentum(momentum=invalid_momentum)


def test_momentum_requires_gradients_before_updating_parameters():
    layer = Dense(2, 2, seed=42)
    optimizer = Momentum(learning_rate=0.1, momentum=0.9)

    with pytest.raises(RuntimeError):
        optimizer.step(layer)